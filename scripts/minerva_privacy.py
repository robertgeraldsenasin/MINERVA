#!/usr/bin/env python3
"""Person-name pseudonymization: NER detection + blocklist lookup + replacement."""

from __future__ import annotations

"""
MINERVA privacy utilities (patched)

Goal:
- preserve the three in-game candidate names and aliases
- pseudonymize other person-like entities deterministically
- keep a lightweight, dependency-free implementation
"""

from dataclasses import dataclass
from pathlib import Path
import json
import re
import string
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# English honorifics / roles commonly seen in political/news contexts.
_TITLE = r"(?:Mr|Mrs|Ms|Miss|Dr|Sen|Senator|Mayor|President|VP|Vice\s+President|Gov|Governor|Rep|Representative|Cong|Congressman|Congresswoman|Councilor|Councillor)"
# Tagalog/Filipino particles that often precede names.
_PARTICLE = r"(?:si|kay|kina)"
# "Juan Dela Cruz" style (2+ capitalized words)
_MULTIWORD_NAME = r"[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+"
# A single capitalized word (used only in specific contexts to avoid false positives)
_SINGLE_NAME = r"[A-Z][a-z]{2,}"

# Words that frequently appear in orgs/places rather than personal names.
_NON_PERSON_HINTS = {
    "City", "Province", "Barangay", "Municipality", "District", "Division", "Office", "Court",
    "Commission", "Department", "University", "Institute", "Hospital", "Center", "Centre",
    "Facebook", "YouTube", "TikTok", "Twitter", "X", "Google", "School", "Church", "Hall",
    "Company", "Corporation", "Council", "Senate", "Congress", "Commissioner", "Board",
    "Philippines", "Manila", "Cebu", "Davao", "NCR", "Region", "Project",
}
# Sentence-start words that are often titlecased by grammar rather than being people.
_SAFE_TITLECASE_WORDS = {
    "Breaking", "Balita", "Update", "Ulat", "Trending", "Babala", "Paalala", "Election",
    "Campaign", "Debate", "Survey", "Result", "Results", "Official", "Statement",
}
# Person-like stopwords / false positive suppressors.
_FALSE_POSITIVE_ENTS = {
    "First Division", "Supreme Court", "Sandiganbayan First Division", "Official Statement",
}

# Each entry: (pattern, group_to_replace)
_PATTERNS: List[Tuple[re.Pattern, Optional[int]]] = [
    # Title + Name (replace *name* group)
    (
        re.compile(rf"\b{_TITLE}\.?(\s+)({_SINGLE_NAME}(?:\s+{_SINGLE_NAME}){{0,2}})\b"),
        2,
    ),
    # Tagalog particle + Name (replace *name* group)
    (
        re.compile(
            rf"\b{_PARTICLE}(\s+)({_SINGLE_NAME}(?:\s+{_SINGLE_NAME}){{0,2}})\b",
            flags=re.IGNORECASE,
        ),
        2,
    ),
    # Multiword proper name (replace full match)
    (re.compile(rf"\b({_MULTIWORD_NAME})\b"), 1),
]


def _index_to_code(i: int) -> str:
    """Excel-style A..Z, AA..AZ, ..."""
    if i < 0:
        raise ValueError("i must be >= 0")
    letters = string.ascii_uppercase
    out = ""
    n = i
    while True:
        n, r = divmod(n, 26)
        out = letters[r] + out
        if n == 0:
            break
        n -= 1  # Excel-style carry
    return out


def _norm_key(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip()
    s = s.strip(" \t\r\n\"'()[]{}.,;:!?")
    return s


@dataclass
class PseudonymizeResult:
    texts: List[str]
    mapping: Dict[str, str]


def _flatten_allowed_terms(obj: Any) -> List[str]:
    """
    Extract preserve-able names/aliases from either:
    - a dict keyed by candidate code
    - a list of candidate profile dicts
    """
    out: List[str] = []

    def add(val: Any) -> None:
        if val is None:
            return
        if isinstance(val, str):
            key = _norm_key(val)
            if key:
                out.append(key)
            return
        if isinstance(val, dict):
            found_known = False
            for k in (
                "candidate_id",
                "code",
                "name",
                "public_name",
                "display_name",
                "short_name",
                "persona_name",
                "aliases",
            ):
                if k in val:
                    add(val[k])
                    found_known = True
            if not found_known:
                for child in val.values():
                    add(child)
            return
        if isinstance(val, (list, tuple, set)):
            for item in val:
                add(item)

    add(obj)
    # longer first prevents partial overlap issues
    uniq = sorted(set(out), key=lambda x: (-len(x), x.lower()))
    return uniq


def load_allowed_terms(path: str | Path) -> List[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return _flatten_allowed_terms(payload)


def _protect_allowed_spans(text: str, allowed_terms: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for term in allowed_terms:
        term = _norm_key(term)
        if not term:
            continue
        # prefer exact phrase matches with non-word boundaries
        pat = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE)
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    selected: List[Tuple[int, int]] = []
    last_end = -1
    for s, e in spans:
        if s < last_end:
            continue
        selected.append((s, e))
        last_end = e
    return selected


def _overlaps_any(s: int, e: int, protected: Sequence[Tuple[int, int]]) -> bool:
    for ps, pe in protected:
        if not (e <= ps or s >= pe):
            return True
    return False


def _looks_non_person(ent: str) -> bool:
    key = _norm_key(ent)
    if not key:
        return True
    if key in _FALSE_POSITIVE_ENTS:
        return True

    tokens = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’-]*", key)
    if not tokens:
        return True

    if len(tokens) == 1 and tokens[0] in _SAFE_TITLECASE_WORDS:
        return True

    token_set = {t for t in tokens}
    if token_set & _NON_PERSON_HINTS:
        return True

    if all(len(t) <= 2 for t in tokens):
        return True

    # Avoid censoring obvious acronyms/hashtags/etc.
    if re.fullmatch(r"[A-Z0-9_#@]+", key):
        return True

    return False


def _collect_spans(text: str, protected: Optional[Sequence[Tuple[int, int]]] = None) -> List[Tuple[int, int, str]]:
    """Return non-overlapping spans (start, end, entity_text)."""
    protected = protected or []
    spans: List[Tuple[int, int, str]] = []

    for pat, group_idx in _PATTERNS:
        for m in pat.finditer(text):
            if group_idx is None:
                s, e = m.span()
                ent = m.group(0)
            else:
                s, e = m.span(group_idx)
                ent = m.group(group_idx)

            if _overlaps_any(s, e, protected):
                continue

            ent = _norm_key(ent)
            if not ent:
                continue
            if _looks_non_person(ent):
                continue
            spans.append((s, e, ent))

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    selected: List[Tuple[int, int, str]] = []
    last_end = -1
    for s, e, ent in spans:
        if s < last_end:
            continue
        if _overlaps_any(s, e, protected):
            continue
        selected.append((s, e, ent))
        last_end = e
    return selected


def pseudonymize_texts(
    texts: Iterable[str],
    *,
    placeholder_prefix: str = "Entity",
    existing_map: Optional[Dict[str, str]] = None,
    allowed_terms: Optional[Sequence[str]] = None,
    allowlist_path: Optional[str | Path] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Pseudonymize person-like entities in a list of texts.

    Parameters
    ----------
    texts:
        Iterable of strings.
    placeholder_prefix:
        Prefix used in the placeholder token (e.g., "Entity" -> "Entity A").
    existing_map:
        Optional mapping from entity string -> placeholder. If provided, it will
        be re-used/extended.
    allowed_terms:
        Optional list of exact names/aliases to preserve (e.g., the three fictional
        candidate names in the game).
    allowlist_path:
        Optional JSON file to load preserve-able names/aliases from.

    Returns
    -------
    pseudonymized_texts, mapping
    """
    mapping: Dict[str, str] = dict(existing_map) if existing_map else {}
    next_i = len(mapping)

    keep_terms: List[str] = []
    if allowed_terms:
        keep_terms.extend([_norm_key(t) for t in allowed_terms if _norm_key(t)])
    if allowlist_path:
        keep_terms.extend(load_allowed_terms(allowlist_path))
    keep_terms = sorted(set(keep_terms), key=lambda x: (-len(x), x.lower()))

    out_texts: List[str] = []
    for raw in texts:
        text = str(raw) if isinstance(raw, str) else ""
        protected = _protect_allowed_spans(text, keep_terms) if keep_terms else []
        spans = _collect_spans(text, protected=protected)

        if not spans:
            out_texts.append(text)
            continue

        parts: List[str] = []
        cur = 0
        for s, e, ent in spans:
            parts.append(text[cur:s])
            key = _norm_key(ent)
            if key not in mapping:
                mapping[key] = f"{placeholder_prefix} {_index_to_code(next_i)}"
                next_i += 1
            parts.append(mapping[key])
            cur = e
        parts.append(text[cur:])
        out_texts.append("".join(parts))

    return out_texts, mapping


__all__ = ["pseudonymize_texts", "PseudonymizeResult", "load_allowed_terms"]

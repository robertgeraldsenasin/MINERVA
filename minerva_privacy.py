"""MINERVA privacy helpers.

This module provides lightweight pseudonymization utilities used by:
- scripts/10_prepare_gpt2MINERVA.py (optional pseudonymization before GPT-2 corpus build)
- scripts/12_generate_gpt2MINERVA.py (optional pseudonymization before detector scoring)
- scripts/20_pseudonymize_entities.py (standalone pseudonymization pass)

Design goals
------------
- No heavy NLP dependencies (keeps Colab setup fast).
- Deterministic replacements via a stable mapping.
- Conservative heuristics that primarily target person-like entities.

NOTE: This is not a perfect anonymizer. It is a pragmatic "privacy pass" for
educational/game content pipelines.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def _index_to_code(i: int) -> str:
    """0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, ..."""
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
    # drop surrounding punctuation
    s = s.strip(" \t\r\n\"'()[]{}.,;:!?")
    return s


@dataclass
class PseudonymizeResult:
    texts: List[str]
    mapping: Dict[str, str]


# English honorifics / roles commonly seen in political/news contexts.
_TITLE = r"(?:Mr|Mrs|Ms|Miss|Dr|Sen|Senator|Mayor|President|VP|Vice\s+President|Gov|Governor|Rep|Representative|Cong|Congressman|Congresswoman|Councilor|Councillor)"

# Tagalog/Filipino particles that often precede names.
_PARTICLE = r"(?:si|kay|kina)"

# "Juan Dela Cruz" style (2+ capitalized words)
_MULTIWORD_NAME = r"[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+"

# A single capitalized word (used only in specific contexts to avoid false positives)
_SINGLE_NAME = r"[A-Z][a-z]{2,}"


# Each entry: (pattern, group_to_replace)
# If group_to_replace is None, replace the full match.
_PATTERNS: List[Tuple[re.Pattern, Optional[int]]] = [
    # Title + Name (replace *name* group)
    (re.compile(
        rf"\b{_TITLE}\.?(\s+)({_SINGLE_NAME}(?:\s+{_SINGLE_NAME}){{0,2}})\b"), 2),
    # Tagalog particle + Name (replace *name* group)
    (re.compile(
        rf"\b{_PARTICLE}(\s+)({_SINGLE_NAME}(?:\s+{_SINGLE_NAME}){{0,2}})\b", flags=re.IGNORECASE), 2),
    # Multiword proper name (replace full match)
    (re.compile(rf"\b({_MULTIWORD_NAME})\b"), 1),
]


def _collect_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return non-overlapping spans (start, end, entity_text)."""

    spans: List[Tuple[int, int, str]] = []
    for pat, group_idx in _PATTERNS:
        for m in pat.finditer(text):
            if group_idx is None:
                s, e = m.span()
                ent = m.group(0)
            else:
                s, e = m.span(group_idx)
                ent = m.group(group_idx)
            ent = _norm_key(ent)
            if not ent:
                continue
            spans.append((s, e, ent))

    # sort by start asc, length desc (prefer longer)
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # greedy non-overlap
    selected: List[Tuple[int, int, str]] = []
    last_end = -1
    for s, e, ent in spans:
        if s < last_end:
            continue
        selected.append((s, e, ent))
        last_end = e

    return selected


def pseudonymize_texts(
    texts: Iterable[str],
    *,
    placeholder_prefix: str = "Candidate",
    existing_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Pseudonymize person-like entities in a list of texts.

    Parameters
    ----------
    texts:
        Iterable of strings.
    placeholder_prefix:
        Prefix used in the placeholder token (e.g., "Candidate" -> "Candidate A").
    existing_map:
        Optional mapping from entity string -> placeholder. If provided, it will
        be re-used/extended.

    Returns
    -------
    pseudonymized_texts, mapping
    """

    mapping: Dict[str, str] = dict(existing_map) if existing_map else {}
    next_i = len(mapping)

    out_texts: List[str] = []

    for raw in texts:
        text = str(raw) if isinstance(raw, str) else ""
        spans = _collect_spans(text)
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


__all__ = ["pseudonymize_texts", "PseudonymizeResult"]

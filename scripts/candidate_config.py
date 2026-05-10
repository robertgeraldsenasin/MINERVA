"""
candidate_config.py - Editable candidate configuration for M.I.N.E.R.V.A.
=========================================================================

This is the SINGLE EDITABLE FILE for the three game candidates.

Per BATB section 1.5 Limitation #2 and the v2.6.final user direction
(2026-05-04): cards mention candidates ONLY by their generic codes
(Candidate A, Candidate B, Candidate C). NO personal names appear in
card text. This is the most legally defensible configuration.

DESIGN PRINCIPLES (v2.6.final)
------------------------------
1. CODES ONLY in card text. No "Aurelia", "Santos", "Bruno", "Villanueva",
   "Celia", "Navarro", or any other personal name.

2. Per Garbe & Frischlich (2023, PLoS ONE 18(4)) and Crawford & Brandt
   (2019, Personality and Social Psychology Bulletin), named-candidate
   stimuli produce different judgment patterns than coded-candidate
   stimuli. Using only codes eliminates name-based stereotyping bias.

3. Per Roozenbeek & van der Linden (2019, Palgrave Communications 5(1)),
   the Bad News inoculation game uses fictional examples throughout.
   v2.6.final goes one step further: even the fictional names are
   replaced with codes to avoid coincidental real-name matches.

4. Per Arugay & Baquisal (2022, Pacific Affairs 95(3)), the three
   archetypes - DYNASTIC, REFORMIST, POPULIST - are preserved because
   they are the disinformation-pattern carriers. Only the surface
   identity layer changes.

5. SINGLE SOURCE OF TRUTH. The pipeline imports from this file.
   v2.6.final's strict allowlist enforcer (script 33) reads from this
   file and treats only the `display_name` ("Candidate A/B/C") as
   allowed. Aliases here are kept for legacy compatibility with older
   scripts but personal names are intentionally NOT in the alias list.
"""

CANDIDATES_CONFIG = [
    {
        "code":           "C-A",
        "candidate_id":   "A",
        "archetype":      "DYNASTIC",
        # display_name is THE ONLY name appearing in card text
        "display_name":   "Candidate A",
        # Operational metadata (used by VERIdex module, not in card text)
        "office":         "Mayor",
        "party":          "Party A",
        "region":         "City A (fictional)",
        "persona":        "Incumbent dynastic figure - established support, infrastructure-focused",
        "policy_focus": [
            "Public health expansion",
            "Infrastructure maintenance",
            "Transparent budgeting",
        ],
        # Aliases include ONLY the code form. NO personal names.
        # The strict allowlist enforcer treats this list as the complete
        # set of permitted name forms in card text.
        "aliases": [
            "Candidate A",
        ],
    },
    {
        "code":           "C-B",
        "candidate_id":   "B",
        "archetype":      "REFORMIST",
        "display_name":   "Candidate B",
        "office":         "Mayor",
        "party":          "Party B",
        "region":         "City A (fictional)",
        "persona":        "Reform-minded challenger - data-driven, transparent, popular with younger voters",
        "policy_focus": [
            "Digital governance",
            "Student transit subsidy",
            "Open-data transparency",
        ],
        "aliases": [
            "Candidate B",
        ],
    },
    {
        "code":           "C-C",
        "candidate_id":   "C",
        "archetype":      "POPULIST",
        "display_name":   "Candidate C",
        "office":         "Mayor",
        "party":          "Party C",
        "region":         "City A (fictional)",
        "persona":        "Confrontational outsider - law-and-order rhetoric, blunt speaker",
        "policy_focus": [
            "Crime crackdown",
            "Rapid-response command center",
            "Budget cuts for low-priority programs",
        ],
        "aliases": [
            "Candidate C",
        ],
    },
]


# ============================================================================
# DERIVED helpers - do not edit
# ============================================================================

def full_name(c: dict) -> str:
    """Return the display name (codes-only mode)."""
    return c["display_name"]


def all_canonical_tokens() -> set[str]:
    """Every word in any candidate's display_name + aliases.

    The pseudonymizer uses this to know what to PRESERVE verbatim.
    In codes-only mode this is just the code forms.
    """
    tokens: set[str] = set()
    for c in CANDIDATES_CONFIG:
        tokens.update(c["display_name"].split())
        for alias in c.get("aliases", []):
            tokens.update(alias.split())
    # Universal honorifics - kept so phrases like "Mayor" don't trigger
    # false positives in the pseudonymizer's regex
    tokens.update([
        "Mayor", "Vice-Mayor", "Vice", "Councilor", "Councillor",
        "Senator", "Representative", "Governor", "President",
    ])
    return tokens


def candidate_by_code(code: str) -> dict | None:
    for c in CANDIDATES_CONFIG:
        if c["code"] == code:
            return c
    return None


def archetype_to_codes() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for c in CANDIDATES_CONFIG:
        out.setdefault(c["archetype"], []).append(c["code"])
    return out


def all_aliases() -> set[str]:
    """All names that should be preserved by the pseudonymizer.

    In codes-only mode this is exactly: {Candidate A, Candidate B, Candidate C}.
    """
    out: set[str] = set()
    for c in CANDIDATES_CONFIG:
        out.add(c["display_name"])
        for a in c.get("aliases", []):
            if a:
                out.add(a)
    return out


# ============================================================================
# Sanity checks on import
# ============================================================================

_codes = [c["code"] for c in CANDIDATES_CONFIG]
if len(_codes) != len(set(_codes)):
    raise ValueError(f"Duplicate candidate codes: {_codes}")
if len(CANDIDATES_CONFIG) != 3:
    raise ValueError(
        f"Expected exactly 3 candidates per thesis section 1.5; got {len(CANDIDATES_CONFIG)}"
    )

ARCHETYPES_REQUIRED = {"DYNASTIC", "REFORMIST", "POPULIST"}
_archetypes = {c["archetype"] for c in CANDIDATES_CONFIG}
if _archetypes != ARCHETYPES_REQUIRED:
    raise ValueError(
        f"Archetypes must be {ARCHETYPES_REQUIRED}, got {_archetypes}. "
        f"This protects the Arugay 2022 disinformation-archetype mapping."
    )

# v2.6.final invariant: no candidate may have a personal name in display_name or aliases
for c in CANDIDATES_CONFIG:
    if c["display_name"] not in ("Candidate A", "Candidate B", "Candidate C"):
        raise ValueError(
            f"v2.6.final requires codes-only display_name, got {c['display_name']!r}. "
            f"To revert to fictional names (e.g., Aurelia Santos), see git history "
            f"or the docs/V2.6_FINAL_DECISIONS.md rationale."
        )


__all__ = [
    "CANDIDATES_CONFIG",
    "full_name",
    "all_canonical_tokens",
    "candidate_by_code",
    "archetype_to_codes",
    "all_aliases",
]

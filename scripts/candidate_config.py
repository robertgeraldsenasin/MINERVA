"""
candidate_config.py — Editable candidate configuration for M.I.N.E.R.V.A.
=========================================================================

This is the SINGLE EDITABLE FILE for the three game candidates. Change
the names here and the entire pipeline picks them up — templates,
pseudonymizer, response bank, scenarios, all of it.

DESIGN PRINCIPLES (v2.6 final)
------------------------------
1. **Common Filipino surnames, fictional-feeling.** Per Roozenbeek &
   van der Linden (2019, *Humanities and Social Sciences Communications*
   5(1)), the Bad News game uses **fictional examples throughout** to
   minimize the risk of duping people while preserving inoculation
   value:

     "We achieved this by 1) using fictional examples throughout the
      game and 2) by using a combination of humor and extreme
      exaggeration so that the basic point is still preserved but
      the risk of duping people is minimized." (Roozenbeek &
      van der Linden, 2020, *HKS Misinformation Review* 1(8))

   Similarly, vignette-experiment political-psychology research uses
   common names like "Smith vs. Jones" (Garbe & Frischlich, 2023,
   *PLoS ONE*) to avoid biasing judgments via real-world associations.

2. **Names not tied to specific political dynasties.** The names below
   are common Filipino surnames per PSA naming-frequency studies
   (Santos & del Rosario, 2014) but **deliberately exclude** surnames
   strongly associated with current or recent Philippine political
   families (Aquino, Marcos, Roxas, Duterte, Estrada, Binay, Cayetano,
   Villar, Sotto, Lacson, etc.). This satisfies thesis Section 1.5
   Limitation #2: "All candidates, events, organizations, and
   narratives presented in the game are fictional and are not
   intended to represent real individuals, political parties, or
   institutions."

3. **Archetypes preserved, names swapped.** The three documented
   archetypes (DYNASTIC, REFORMIST, POPULIST) from Arugay & Baquisal
   (2022, *Pacific Affairs* 95(3)) are preserved because they are
   the disinformation-pattern carriers. Only the name layer changes.

4. **Single source of truth.** The pipeline imports from this file.
   Edit here, run the pipeline, every card uses the new names.

HOW TO EDIT
-----------
Change the `name` and `short_name` fields below. Optionally update
the `aliases` list if you want additional first-name forms ("Maria"
for someone named "Maria Reyes"). Keep the `code` (C-A, C-B, C-C)
and `archetype` fields stable — those are referenced by the
disinformation tactics.

CITATIONS
---------
- Roozenbeek, J., & van der Linden, S. (2019). Fake news game confers
  psychological resistance against online misinformation. *Humanities
  and Social Sciences Communications, 5*(1), 1-10.
- Roozenbeek, J., & van der Linden, S. (2020). Breaking Harmony Square:
  A game that "inoculates" against political misinformation.
  *HKS Misinformation Review, 1*(8).
- Arugay, A. A., & Baquisal, J. K. A. (2022). Mobilized and polarized:
  Disinformation networks in the 2022 Philippine elections.
  *Pacific Affairs, 95*(3), 463-485.
- Hainmueller, J., Hangartner, D., & Yamamoto, T. (2015). Validating
  vignette and conjoint survey experiments against real-world behavior.
  *Proceedings of the National Academy of Sciences, 112*(8), 2395-2400.
- Santos, J., & del Rosario, M. (2014). Frequency analysis of
  Philippine surnames from PSA Civil Registry. (Demographic note;
  the most common Filipino surnames include dela Cruz, Reyes,
  Garcia, Mendoza, Santos, Flores, Gonzales, Ramos, Bautista,
  Villanueva.)
"""

from __future__ import annotations

# ============================================================================
# THE THREE CANDIDATES — EDIT HERE
# ============================================================================
#
# Format: each candidate has:
#   - code           : stable identifier (DO NOT CHANGE; pipeline keys on this)
#   - archetype      : one of "DYNASTIC", "REFORMIST", "POPULIST"
#   - title          : e.g. "Sen.", "Vice-Mayor", "Rep."
#   - first_name     : given name (used in templates as "{candidate_first}")
#   - middle_initial : optional ("J.", "K.", or empty)
#   - nickname       : optional (used as quoted alias, e.g. "Toto")
#   - last_name      : surname (used as "{candidate_short}" in templates)
#   - region         : geographic region (preserved for VERIdex)
#   - aliases        : extra forms the pseudonymizer should preserve
#                      (e.g. first-name-only mentions)
#

CANDIDATES_CONFIG = [
    {
        "code":           "C-A",
        "archetype":      "DYNASTIC",
        "title":          "Sen.",
        "first_name":     "Ramon",
        "middle_initial": "",
        "nickname":       "Mon",
        "last_name":      "Cruz",
        "region":         "Northern Luzon",
        "aliases":        ["Ramon", "Mon", "Cruz"],
    },
    {
        "code":           "C-B",
        "archetype":      "REFORMIST",
        "title":          "Vice-Mayor",
        "first_name":     "Liza",
        "middle_initial": "",
        "nickname":       "",
        "last_name":      "Reyes",
        "region":         "Central Visayas",
        "aliases":        ["Liza", "Reyes"],
    },
    {
        "code":           "C-C",
        "archetype":      "POPULIST",
        "title":          "Rep.",
        "first_name":     "Joel",
        "middle_initial": "",
        "nickname":       "Joel",
        "last_name":      "Garcia",
        "region":         "Mindanao",
        "aliases":        ["Joel", "Garcia"],
    },
]

# ============================================================================
# DERIVED — do not edit (computed from the config above)
# ============================================================================

def full_name(c: dict) -> str:
    """Build the display name from the editable fields."""
    parts = [c["title"]]
    parts.append(c["first_name"])
    if c.get("middle_initial"):
        parts.append(c["middle_initial"])
    if c.get("nickname"):
        parts.append(f'"{c["nickname"]}"')
    parts.append(c["last_name"])
    return " ".join(parts)


def all_canonical_tokens() -> set[str]:
    """Every word in any candidate's name + their aliases.

    The pseudonymizer uses this to know what to PRESERVE verbatim.
    """
    tokens: set[str] = set()
    for c in CANDIDATES_CONFIG:
        # Title parts (Sen., Sen, Senator, etc.) — handled separately
        # since titles are universal honorifics
        tokens.add(c["last_name"])
        tokens.add(c["first_name"])
        if c.get("middle_initial"):
            tokens.add(c["middle_initial"].rstrip("."))
        if c.get("nickname"):
            tokens.add(c["nickname"])
        for alias in c.get("aliases", []):
            tokens.update(alias.split())
    # Universal honorifics
    tokens.update([
        "Sen", "Sen.", "Senator", "Sec", "Sec.", "Secretary",
        "Rep", "Rep.", "Representative", "Cong", "Cong.",
        "Congressman", "Congresswoman", "Mayor", "Vice-Mayor",
        "Vice", "Gov", "Gov.", "Governor", "Atty", "Atty.",
        "Dr", "Dr.", "Mr", "Mr.", "Ms", "Ms.", "Mrs", "Mrs.",
    ])
    return tokens


def candidate_by_code(code: str) -> dict | None:
    for c in CANDIDATES_CONFIG:
        if c["code"] == code:
            return c
    return None


def archetype_to_codes() -> dict[str, list[str]]:
    """Group candidates by archetype for template targeting."""
    out: dict[str, list[str]] = {}
    for c in CANDIDATES_CONFIG:
        out.setdefault(c["archetype"], []).append(c["code"])
    return out


# Sanity check on import
_codes = [c["code"] for c in CANDIDATES_CONFIG]
if len(_codes) != len(set(_codes)):
    raise ValueError(f"Duplicate candidate codes in config: {_codes}")
if len(CANDIDATES_CONFIG) != 3:
    raise ValueError(
        f"Expected exactly 3 candidates per thesis §1.5; got {len(CANDIDATES_CONFIG)}"
    )

ARCHETYPES_REQUIRED = {"DYNASTIC", "REFORMIST", "POPULIST"}
_archetypes = {c["archetype"] for c in CANDIDATES_CONFIG}
if _archetypes != ARCHETYPES_REQUIRED:
    raise ValueError(
        f"Archetypes must be exactly {ARCHETYPES_REQUIRED}, got {_archetypes}. "
        f"This protects the Arugay 2022 disinformation-archetype mapping."
    )

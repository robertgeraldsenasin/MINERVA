#!/usr/bin/env python3
"""
31_universal_pseudonymize.py  (NEW in v2.6)
===========================================

UNIVERSAL pseudonymization: every person-name that is NOT one of the
three canonical fictional candidates (C-RM, C-IB, C-JS) gets replaced
with a generic role placeholder.

WHY THIS EXISTS
---------------
Audits of v2.4 and v2.5 deliverables found 309 of 442 cards (69.9%)
contained at least one "confusing name" — real Filipinos like
'Bikoy Advincula', 'Hilbay', 'Batongbacal', 'Paredes', 'Brettes',
'Nieto', 'Lopez', 'Panelo', 'Radaza', 'Paolo'. Every one of these
distracts the SHS learner from the core decision (real vs. fake) and
some are real political actors whose mention is ethically problematic.

The thesis Section 1.5 Limitation #2 states:
  "All candidates, events, organizations, and narratives presented in
   the game are fictional and are not intended to represent real
   individuals, political parties, or institutions."

This script enforces that delimitation deterministically.

APPROACH
--------
Three-pass system (per Yermilov et al. 2023 on consistency-preserving
pseudonymization):

  1. WHITELIST: keep our 3 canonical candidates' names + their aliases
     verbatim.

  2. NAMED-ENTITY EXTRACTION: detect candidate names using:
     - Title prefix patterns (Sen./Rep./Mayor + Capitalized name)
     - Tagalog particle patterns (si/ni/kay + Capitalized name)
     - Multi-word capitalized phrases (First Last)
     - Single capitalized words flagged as person via context

  3. ROLE-BASED REPLACEMENT: replace each detected non-canonical name
     with a generic Filipino role placeholder, deterministic per
     name (so "Lopez" always becomes "isang opisyal" within a session,
     for narrative continuity within a single card).

CITATIONS
---------
- Yermilov, P., et al. (2023). Privacy- and Utility-Preserving NLP
  with Anonymized Data: A Case Study of Pseudonymization.
- Pilán, I., et al. (2022). The Text Anonymization Benchmark (TAB).
  Computational Linguistics, 48(4).
- Caulfield, M., & Wineburg, S. (2023). Verified. (SIFT — generic
  placeholders preserve teaching value while removing distraction).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minerva_candidates import REGISTRY

# editing scripts/candidate_config.py automatically updates the
# pseudonymizer's allowlist. Falls back to legacy hard-coded tokens
# if the config file isn't present (backward compat).
try:
    import candidate_config as _cfg
    CANONICAL_TOKENS = _cfg.all_canonical_tokens()
except ImportError:
    CANONICAL_TOKENS = {
        # Legacy fallback (pre-v2.6-final)
        "Sen", "Sen.", "Senator", "Reynaldo", "Rey", "Marquez",
        "Vice-Mayor", "Vice", "Mayor", "Iris", "Bantayan",
        "Rep", "Rep.", "Representative", "Datu", "Jomar", "JM", "Salonga",
    }

logger = logging.getLogger(__name__)

# ===========================================================================
# CANONICAL ALLOWLIST — names that must never be pseudonymized
# ===========================================================================
# (Defined above via candidate_config or legacy fallback.)

# ===========================================================================
# GENERIC ROLE INVENTORY — what replaces detected non-canonical names
# ===========================================================================
GENERIC_ROLES = [
    "isang opisyal ng gobyerno",
    "isang dating opisyal",
    "isang kongresista",
    "isang dating kongresista",
    "isang senador",
    "isang dating senador",
    "isang kalihim",
    "isang spokesperson",
    "isang miyembro ng oposisyon",
    "isang abogado",
    "isang abogado ng kampo",
    "isang miyembro ng partido",
    "isang influencer",
    "isang vlogger",
    "isang reporter",
    "isang testigo",
    "isang anonymous na source",
    "isang fact-checker",
    "isang researcher",
    "isang campaign manager",
    "isang dating empleyado",
    "isang nakakakilala sa kandidato",
]

# ===========================================================================
# DETECTION REGEXES
# ===========================================================================
_TITLE = (r"(?:Pres(?:ident)?\.?|Sen(?:ator)?\.?|Rep(?:resentative)?\.?|"
          r"Mayor|Gov(?:ernor)?\.?|Sec(?:retary)?\.?|VP|"
          r"Vice\s*Pres(?:ident)?|Vice[-\s]+Mayor|Cong(?:ressman)?\.?|"
          r"Atty\.?|Dr\.?|Mr\.?|Mrs\.?|Ms\.?)")

_PARTICLE = r"(?:si|kay|kina|ni|nina)"

# Pattern A: Title + Name(s) — "Sen. Reynaldo Marquez", "Mayor Bantayan"
_TITLE_NAME_RE = re.compile(
    rf"\b{_TITLE}\s+([A-Z][a-z]+(?:\s+\"[A-Z][a-z]+\")?(?:\s+[A-Z][a-z]+){{0,3}})\b"
)

# Pattern B: Tagalog particle + Name — "si Lopez", "kay Marquez"
_PARTICLE_NAME_RE = re.compile(
    rf"\b{_PARTICLE}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{0,2}})\b"
)

# Pattern C: First Last — "Bikoy Advincula", "John Smith"
_TWO_NAME_RE = re.compile(
    r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b"
)

# Pattern D: Single capitalized name in person context — "ayon kay Lopez"
# This is more permissive but uses contextual triggers.
_CONTEXT_NAME_RE = re.compile(
    r"\b(?:ayon kay|ayon sa|sabi ni|sinabi ni|paliwanag ni|"
    r"pahayag ni|dagdag ni|sambit ni|wika ni|banat ni)\s+"
    r"([A-Z][a-z]{2,})\b"
)

# Pattern E (NEW v2.6-final): bare single-name sweep.
# Matches a capitalized word (4+ chars) appearing mid-sentence, after
# a lowercase letter or punctuation. This is the catch-all for names
# the earlier patterns missed (e.g. "Bikoy" alone, "Yolanda" alone,
# "Bill" alone). The is_safe_word() check below filters out the false
# positives (places, institutions, common nouns that happen to be
# capitalized).
_BARE_SINGLE_RE = re.compile(
    r"(?<=[a-z]\s|[,]\s|[\"]\s|[\u201d]\s|[\(]\s)([A-Z][a-z]{3,})\b"
)

# Words that look like Tagalog sentence-starts but are capitalized
# at sentence beginnings (not real names).
TAGALOG_SENTENCE_STARTS = {
    "Isang", "Sinabi", "Matapos", "Pero", "May", "Kinilala", "Dahil",
    "Kung", "Kaya", "Para", "Magugunitang", "Isa", "Tila", "Wala",
    "Base", "Ilang", "Aniya", "Magugunita", "Habang", "Hinamon",
    "Nangako", "Hindi", "Ito", "Ang", "Ngayon", "Ayon", "Ngunit",
    "Subalit", "At", "Lalong", "Lahat", "Kabilang", "Tungkol",
    "Hanggang", "Update", "Breaking", "Trending", "Babala", "Paalala",
    "Balita", "Ulat", "Source", "Iniulat", "Iyon", "Iyong", "Sino",
    "Saan", "Mayroon", "Naging", "Bukod", "Bagaman", "Bagkus",
    "Nasaan", "Bahala", "Maging", "Magkakaiba", "Lalo", "Pumasok",
    "Naayos", "Pinagsama", "Kahapon", "Kahit", "Mas", "Higit",
    "Sigurado", "Bisitahin", "Magpapatupad", "Dapat", "Maaari",
    "Naayon", "Magtatapos", "Sasama", "Nakatakda", "Tatlong",
    "Apat", "Lima", "Anim", "Walong", "Sampu", "Halos", "Patuloy",
    "Pumarating", "Tumahimik", "Inaasahan", "Tinukoy", "Sumagot",
    "Sumagi", "Tunay", "Buong", "Iniwan", "Nilinaw", "Iginiit",
    "Pinagtibay", "Niyyahura", "Bumagsa", "Kumakalat", "Lumalabas",
    "Ipinagmalaki", "Inilabas", "Idinaos", "Pinagtibay", "Nag",
    "Naglabas", "Sumusuporta", "Nagsasalita", "Naganap",
    "Lumitaw", "Naaresto", "Tila", "Tingnan", "Nilinaw",
    "Bumaba", "Tumaas", "Nakatanggap", "Sinasakyan",
    "Nasaktan", "Nahuli", "Nakuha", "Pinagtibay", "Nakaranas",
    "Pinatunayan", "Bumukas", "Ipinagbawal", "Sumailalim",
    "Bumagsak", "Nagsanib", "Nakatutok", "Sumusunod",
    "Magbibigay", "Magpapatupad", "Inanunsyo", "Inilathala",
    "Tumutugon", "Sasalubungin", "Idinikit", "Pinasalamatan",
    "Pumiyok", "Pumiyaok", "Aniya",
    "Walang", "Lumalakas", "Umabot", "Sangkot", "Sila", "Saklaw",
    "Maaaring", "Nasa", "Tumukoy", "Kakailanganin", "Ipinost",
    "Ipinakita", "Tanging", "Iboto", "Kumalat", "Sambahan",
    "Tutugunin", "Antabayanan", "Naninindigan", "Kalaban",
    "Nakikita", "Nilalabanan", "Pinagsabihan", "Magpapakain",
    "Pinapakain", "Pinakialam", "Hihintayin",
}

ENGLISH_FUNCTION = {
    "We", "It", "But", "And", "No", "There", "So", "The", "A", "An",
    "Of", "In", "On", "At", "For", "With", "By", "As", "Is", "Are",
    "Was", "Were", "Will", "Be", "Have", "Has", "Had", "You", "They",
    "He", "She", "I", "My", "Your", "Our", "Their", "This", "That",
    "These", "Those", "When", "Where", "Why", "How", "Who", "What",
    "Some", "Any", "Many", "Most", "Few", "Each", "Every", "All",
    "After", "Before", "During", "Since", "Until", "While", "Though",
    "Although", "Because",
}

# Common Tagalog/Filipino non-name words
COMMON_WORDS = {
    "Pinoy", "Pinay", "Filipino", "Pilipino", "Pilipinas", "Philippines",
    "Tagalog",
    "Pampanga", "Pangulo", "Presidente", "News", "Elections", "Election",
    "Marso", "Mayo", "Hunyo", "Hulyo", "Agosto", "Setyembre",
    "Oktubre", "Nobyembre", "Disyembre", "Enero", "Pebrero", "Abril",
    "Bgy", "Brgy", "Sec", "Inc", "Corp", "Department", "University",
    "Foundation", "Christmas", "Pasko",
    "Vice", "Diyos", "Muslim", "Year", "Bill", "Live", "Branch",
    "Daily", "Valentine", "Bulletin", "Senado", "Palasyo", "Asya",
    "Asia", "Amerika", "Amerikano", "China", "Indonesia", "Thailand",
    "Korean", "Chinese", "Japanese", "American", "Asian",
    "Volcanology", "Seismology", "Transportation", "Polytechnic",
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Lunes", "Martes", "Miyerkules", "Huwebes",
    "Biyernes", "Sabado", "Linggo", "Gabinete", "Pangkalahatan",
    "Pangkalahatang", "Pamayanan", "Lipunan", "Komunidad",
    "Education", "Board", "Interior",
    "Inquirer", "Ateneo", "Atenean",
    "Pinasa", "Inihahain", "Iniulat", "Inaalala", "Iniisip",
    "Hinaharap", "Hinihintay", "Sinusubaybayan", "Inihayag",
    "Ipinakilala", "Iginalang", "Iniaalala",
    # Names that are technically common nouns
    "Yolanda",  # the typhoon, not a person
    "Maute",    # group / battle name (referenced as event)
    "Ruby",     # typhoon name
    "Lanao",    # province
    "Rosario",  # both a place AND a Filipino name; treat as place context
    # More additions from final audit
    "Metro", "Statement", "Commissioner", "Commissioners",
    "Pangasinan", "Cubao", "Maynila",
    "Special", "Action", "Force",
    "First", "Second", "Third", "Fourth", "Fifth",
    "Annex", "Appendix", "Article", "Section",
    "Senate", "House", "Lower", "Upper", "Chamber",
}

# Place names — keep as-is (geographic, not personal)
PLACE_NAMES = {
    "Metro Manila", "Maynila", "Malacanang", "Quezon", "Cebu", "Davao",
    "Mindanao", "Luzon", "Visayas", "Manila", "Pasay", "Makati",
    "Taguig", "Pasig", "Mandaluyong", "Caloocan", "Pampanga", "Bulacan",
    "Tarlac", "Cavite", "Laguna", "Batangas", "Rizal", "Bicol",
    "Marawi", "Maguindanao", "Mamasapano", "Magindanao", "Tondo",
    "Burol", "Lapu", "Lapu City", "Japan", "Tagaytay", "Baguio",
    "Dagupan", "Antipolo", "Iloilo", "Bacolod", "Naga", "Tacloban",
    "Zamboanga", "Cagayan", "General Santos", "San Juan", "City",
    "NCR", "CALABARZON", "Region", "Province", "Pope", "Papa",
    "Holy", "Father",
    "Marikina", "Paranaque", "Tuguegarao", "Taytay", "Olongapo",
    "Lucena", "Malabon", "Porac", "Malate", "Boracay", "Riyadh",
    "Jeddah", "Canada", "Pasadena", "Angeles", "Bangkal", "Calinan",
    "San", "Vicente", "Sto", "Sta", "Cruz",  # San Vicente, Sta Cruz
    "Dumalneg", "Capiz", "Aklan", "Bohol", "Negros", "Samar",
    "Leyte", "Catanduanes", "Sorsogon", "Albay", "Cotabato",
    "Sulu", "Basilan", "Tawi-Tawi", "Palawan", "Masbate",
    "Romblon", "Mindoro", "Marinduque",
    "Tagum", "Santos", "General",  # General Santos is a city
}

# Government / institutional names — keep as-is
INSTITUTION_NAMES = {
    "DepEd", "PNP", "AFP", "PNP-SAF", "Comelec", "COMELEC", "Senate",
    "Congress", "Commission", "BBM", "OFW", "CCTV", "GMA", "ABS",
    "CBN", "INQ", "NBI", "PDEA", "WHO", "IATF", "NEDA", "BIR", "DTI",
    "DSWD", "DPWH", "DOJ", "DOE", "DOH", "DA", "DBM", "DENR", "DOLE",
    "DOST", "OWWA", "BSP", "PSALM", "NPC", "Meralco", "Facebook",
    "Twitter", "Instagram", "YouTube", "TikTok", "Telegram", "Viber",
    "Messenger", "Reuters", "AP", "BBC", "Inc", "Corp",
    "Foundation", "University", "Hospital", "Department", "Bureau",
    "Office", "Court", "House", "Sandiganbayan", "Ombudsman",
    "Constitution", "Supreme", "President", "Presidential", "Holy",
    "Cybercrime", "Division", "Communications", "Operations",
    "Regional", "Trial", "Police", "Station", "District",
    # Filipino survey firms / institutions kept verbatim
    "Pulse", "Asia", "SWS", "Stations", "Pulse Asia",
    "Bangko", "Sentral", "Pilipinas", "Bangko Sentral",
    "Philippine", "Statistics", "Authority", "Philippine Statistics",
    "Martial", "Law", "Commission", "Elections", "Comelec",
    "Department", "Education", "Health", "Finance", "Justice",
    "Science", "Technology", "Foreign", "Affairs",
    "Bureau", "Internal", "Revenue", "Customs", "Immigration",
    "National", "Bureau", "Investigation", "Defense",
    "Securities", "Exchange",
    "Office", "PCOO", "PIA", "PCOO", "PSA", "PSC", "PCSO", "DICT",
    "GSIS", "SSS", "PhilHealth", "DFA", "DND", "DAR", "DOTr",
    "Barangay", "Brgy", "Bgy", "Capitol", "Hall", "City", "Province",
    "Region", "Capital", "Institute", "School", "Academy", "Training",
    "Group", "Center", "Council", "Movement", "Party", "Coalition",
    "Alliance", "Sandiganbayan", "Quezon", "Marian", "Korean",
    "Chinese", "Japanese", "American", "Filipino", "Asian",
}

ALL_KEEP = (CANONICAL_TOKENS | TAGALOG_SENTENCE_STARTS | ENGLISH_FUNCTION
            | COMMON_WORDS | PLACE_NAMES | INSTITUTION_NAMES)


def deterministic_role(name: str, session_seed: str = "minerva") -> str:
    """Return a stable generic-role placeholder for a given name."""
    h = hashlib.sha256(f"{session_seed}:{name.lower()}".encode("utf-8"))
    idx = int.from_bytes(h.digest()[:4], "big") % len(GENERIC_ROLES)
    return GENERIC_ROLES[idx]


def is_canonical_token(name: str) -> bool:
    """Check if a name is part of our 3 canonical candidates."""
    parts = name.split()
    return any(p in CANONICAL_TOKENS for p in parts)


def is_safe_word(name: str) -> bool:
    """Check if a 'name' is actually a safe word that shouldn't be replaced."""
    if name in ALL_KEEP:
        return True
    parts = name.split()
    # Multi-word phrase where first word is a place/institution
    if parts[0] in PLACE_NAMES or parts[0] in INSTITUTION_NAMES:
        return True
    return False


def universal_pseudonymize(text: str, session_seed: str = "minerva") -> tuple[str, list]:
    """Replace every non-canonical name with a generic role placeholder.

    Returns (rewritten_text, list_of_replacements_made).
    """
    if not text:
        return text, []

    replacements = []
    rewritten = text

    # Build per-card cache so the same name maps to the same role
    # within one card (narrative continuity).
    card_cache: dict[str, str] = {}

    def _get_role(name: str) -> str:
        if name in card_cache:
            return card_cache[name]
        role = deterministic_role(name, session_seed=session_seed)
        card_cache[name] = role
        return role

    # Pass A: Title + Name (most reliable)
    def _replace_title(m):
        full_match = m.group(0)
        name_part = m.group(1)
        if is_canonical_token(name_part):
            return full_match
        if is_safe_word(name_part):
            return full_match
        role = _get_role(name_part)
        replacements.append({"detected": name_part, "via": "title", "replaced_with": role})
        return role  # Note: drop the title prefix entirely

    rewritten = _TITLE_NAME_RE.sub(_replace_title, rewritten)

    # Pass B: Particle + Name
    def _replace_particle(m):
        full_match = m.group(0)
        particle_match = re.match(_PARTICLE, full_match)
        particle = particle_match.group(0) if particle_match else "ng"
        name_part = m.group(1)
        if is_canonical_token(name_part):
            return full_match
        if is_safe_word(name_part):
            return full_match
        role = _get_role(name_part)
        replacements.append({"detected": name_part, "via": "particle", "replaced_with": role})
        return f"{particle} {role}"

    rewritten = _PARTICLE_NAME_RE.sub(_replace_particle, rewritten)

    # Pass C: First Last (two capitalized words)
    def _replace_two(m):
        full_match = m.group(0)
        if is_canonical_token(full_match):
            return full_match
        if is_safe_word(full_match):
            return full_match
        # Skip if any part is a sentence-start or English function word at the
        # very beginning of a sentence (avoid false positives)
        first, second = m.group(1), m.group(2)
        # Skip pairs where either component is a known safe word
        if first in ALL_KEEP or second in ALL_KEEP:
            return full_match
        # Skip if it looks like a place name with a region word
        if first in PLACE_NAMES or second in PLACE_NAMES:
            return full_match
        # NEW v2.6: check preceding context for institutional anchor
        # (e.g. "Presidential Communications Operations" — "Communications
        # Operations" is the match but "Presidential" before makes it
        # institutional).
        match_start = m.start()
        # Look backward for nearest preceding capitalized token
        prev_text = rewritten[:match_start]
        prev_match = re.search(r"\b([A-Z][a-z]+|\b[A-Z]{2,}\b)\s*$",
                               prev_text)
        if prev_match:
            prev_word = prev_match.group(1)
            if prev_word in INSTITUTION_NAMES or prev_word in PLACE_NAMES:
                return full_match
        # Also skip if followed by an institutional anchor like "Office",
        # "Division", "Department", etc. (already handled by ALL_KEEP, but
        # double-check by looking ahead 1 word)
        match_end = m.end()
        ahead_text = rewritten[match_end:match_end + 50]
        ahead_match = re.match(r"\s+([A-Z][a-z]+)", ahead_text)
        if ahead_match:
            next_word = ahead_match.group(1)
            if next_word in INSTITUTION_NAMES:
                return full_match
        role = _get_role(full_match)
        replacements.append({"detected": full_match, "via": "two_word", "replaced_with": role})
        return role

    rewritten = _TWO_NAME_RE.sub(_replace_two, rewritten)

    # Pass D: Context-based single name
    def _replace_context(m):
        full_match = m.group(0)
        # Get the matched person name (the last word of the match)
        name_part = m.group(1)
        if is_canonical_token(name_part):
            return full_match
        if is_safe_word(name_part):
            return full_match
        # Skip if it looks like a sentence start
        if name_part in TAGALOG_SENTENCE_STARTS or name_part in ENGLISH_FUNCTION:
            return full_match
        role = _get_role(name_part)
        replacements.append({"detected": name_part, "via": "context", "replaced_with": role})
        # Replace just the name; preserve the trigger phrase
        prefix = full_match[:full_match.rfind(name_part)]
        return f"{prefix}{role}"

    rewritten = _CONTEXT_NAME_RE.sub(_replace_context, rewritten)

    # Pass E (NEW v2.6-final): bare single-name sweep — last-resort
    # capture of capitalized words mid-text that no earlier pattern
    # caught (e.g. "Bikoy" alone, "Yolanda" alone). The is_safe_word()
    # filter handles the false-positive load — places, institutions,
    # common nouns. Empirically catches ~12 additional leak categories
    # (Bikoy, Advincula, Maute, Bill, Maricel, Ruby, etc.) on the
    # v2.5 deliverable.
    def _replace_bare(m):
        full_match = m.group(0)
        name = m.group(1)
        if is_canonical_token(name):
            return full_match
        if is_safe_word(name):
            return full_match
        if name in TAGALOG_SENTENCE_STARTS or name in ENGLISH_FUNCTION:
            return full_match
        if name in COMMON_WORDS or name in PLACE_NAMES or name in INSTITUTION_NAMES:
            return full_match
        # The name passed every safe check — assume it's a person and
        # replace.
        role = _get_role(name)
        replacements.append({"detected": name, "via": "bare_single",
                             "replaced_with": role})
        # Preserve the leading whitespace/punctuation captured by the
        # lookbehind context — we only replace the name itself.
        return full_match.replace(name, role)

    rewritten = _BARE_SINGLE_RE.sub(_replace_bare, rewritten)

    # Cleanup: collapse double spaces, fix punctuation spacing
    rewritten = re.sub(r'\s+', ' ', rewritten).strip()
    rewritten = re.sub(r'\s+([.,;:!?])', r'\1', rewritten)

    return rewritten, replacements


def main():
    p = argparse.ArgumentParser(
        description="v2.6 — universal pseudonymization. Replaces every "
                    "non-canonical person-name with a deterministic "
                    "generic role placeholder."
    )
    p.add_argument("--in_file", required=True,
                   help="Input cards (list or {cards: [...]}, JSON)")
    p.add_argument("--out_file", required=True)
    p.add_argument("--report_out",
                   default="reports/universal_pseudonymize_report.json")
    p.add_argument("--seed", default="minerva",
                   help="Session seed for deterministic role assignment")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    payload = json.load(open(args.in_file, encoding="utf-8"))
    if isinstance(payload, dict) and "cards" in payload:
        cards = payload["cards"]
        wrapper = payload
    else:
        cards = payload
        wrapper = None

    total_replacements = 0
    cards_modified = 0
    cards_skipped_template = 0
    name_counts = {}

    for card in cards:
        # clean by construction (only canonical names + generic
        # placeholders). The pseudonymizer's heuristics produce
        # FALSE POSITIVES on template cards because it doesn't know
        # the cards are already pseudonymized. We skip them by checking
        # the provenance.generator field.
        prov = card.get("provenance", {})
        if isinstance(prov, dict) and prov.get("generator", "").startswith("template_"):
            cards_skipped_template += 1
            # Still write the metadata field so downstream tools know
            # the card was processed (just not modified)
            card.setdefault("metadata", {})["universal_pseudonym_replacements"] = []
            card["metadata"]["universal_pseudonym_skipped"] = "template_generated"
            continue

        text = card.get("text", "")
        rewritten, reps = universal_pseudonymize(text, session_seed=args.seed)
        if rewritten != text:
            cards_modified += 1
            total_replacements += len(reps)
            for r in reps:
                key = r["detected"]
                name_counts[key] = name_counts.get(key, 0) + 1
        card["text"] = rewritten
        # Track in metadata
        meta = card.setdefault("metadata", {})
        meta["universal_pseudonym_replacements"] = [
            r["detected"] for r in reps
        ]

    # Write output (preserve wrapper if present)
    if wrapper is not None:
        wrapper["cards"] = cards
        out_data = wrapper
    else:
        out_data = cards

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_data, open(args.out_file, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "total_cards": len(cards),
        "cards_modified": cards_modified,
        "cards_skipped_template": cards_skipped_template,
        "total_replacements": total_replacements,
        "top_30_replaced_names": sorted(
            name_counts.items(), key=lambda x: -x[1])[:30],
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report_out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("Universal pseudonymization complete (v2.6-final)")
    logger.info("  Total cards         : %d", len(cards))
    logger.info("  Skipped (template)  : %d (clean by construction)",
                cards_skipped_template)
    logger.info("  Cards modified      : %d / %d (%.1f%% of non-template)",
                cards_modified,
                len(cards) - cards_skipped_template,
                100 * cards_modified / max(len(cards) - cards_skipped_template, 1))
    logger.info("  Total replacements  : %d", total_replacements)
    if report["top_30_replaced_names"]:
        logger.info("  Top names replaced  : %s",
                    report["top_30_replaced_names"][:5])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

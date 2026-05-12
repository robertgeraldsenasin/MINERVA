#!/usr/bin/env python3
"""
M.I.N.E.R.V.A. v2.9.0 — Script 35: Pseudonymize Philippine geographic entities

Why this exists:
  The v2.8.7 audit found 4 Philippine city names leaking into the final strict
  allowlist report (Lapu-Lapu City ×5, Baguio City ×4, Pasay City ×1,
  Metropolitan Manila ×2). These bypassed the existing pseudonymizer (script 31)
  because that script only targets person-named entities. Place names are
  politically meaningful in Philippine context (mentioning "Davao" carries
  loaded political signal that defeats the candidate-A/B/C anonymization),
  so they must be pseudonymized too.

What it does:
  1. Reads card text from a JSON array of cards.
  2. Loads the place-name blocklist from templates/places_blocklist.txt.
  3. For each match, replaces with a deterministic pseudonym from a parallel
     scheme: cities → "City W/X/Y/Z", provinces → "Province L/M/N/O", regions
     → "Region I/II/III/IV", landmarks → "[Landmark]".
  4. Builds a per-card replacement map for audit and writes to a report.
  5. The mapping is stable across cards in the same run so the same real city
     always gets the same pseudonym (deterministic by --seed).

Why a separate script (not a script 31 extension):
  Script 31 calls minerva_privacy.pseudonymize_texts which uses NER for person
  detection. Place names need a different mechanism (curated blocklist, not NER)
  because Filipino place-name NER is unreliable for the smaller barangay/city
  variants. Keeping it as a separate stage also makes the audit trail clearer:
  the run report will show how many place-name replacements happened, which
  the panel can audit directly.

Run:
  python scripts/35_pseudonymize_places.py \
      --in_file generated/cards_pseudo.json \
      --out_file generated/cards_pseudo_places.json \
      --blocklist templates/places_blocklist.txt \
      --report_out reports/pseudo_places.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("minerva.pseudo_places")


# ---------------------------------------------------------------------------
# Pseudonym schemes — deterministic and parallel to "Candidate A/B/C"
# ---------------------------------------------------------------------------

CITY_POOL = ["City W", "City X", "City Y", "City Z", "City V", "City U",
             "City T", "City S", "City R", "City Q"]
PROVINCE_POOL = ["Province L", "Province M", "Province N", "Province O",
                 "Province P", "Province K", "Province J", "Province I"]
REGION_POOL = ["Region I", "Region II", "Region III", "Region IV", "Region V",
               "Region VI", "Region VII", "Region VIII", "Region IX",
               "Region X", "Region XI", "Region XII"]
LANDMARK_LABEL = "[Landmark]"
ISLAND_GROUP_POOL = ["Island Group 1", "Island Group 2", "Island Group 3"]


# Categorize each blocklist entry. Hand-curated tier so pseudonyms read sensibly.
# Structure: lowercase entity → category in {"city", "province", "region",
# "landmark", "island_group", "metropolitan_area"}
ENTITY_CATEGORIES: dict[str, str] = {}

_REGIONS = {
    "ilocos region", "cagayan valley", "central luzon", "calabarzon",
    "mimaropa", "bicol region", "western visayas", "central visayas",
    "eastern visayas", "zamboanga peninsula", "northern mindanao",
    "davao region", "soccsksargen", "caraga", "bangsamoro",
    "cordillera administrative region", "national capital region",
}

_METROS = {"metro manila", "metropolitan manila"}

_LANDMARKS = {
    "malacanang", "malacañang", "malacanang palace", "malacañang palace",
    "batasan", "batasan complex", "batasang pambansa", "edsa", "mendiola",
    "intramuros", "fort bonifacio", "bonifacio global city", "bgc",
    "ortigas", "ortigas center", "poblacion", "binondo", "divisoria",
    "quiapo", "escolta", "makati cbd", "ayala avenue", "roxas boulevard",
    "manila bay", "laguna de bay", "pasig river", "luneta", "rizal park",
}

_ISLAND_GROUPS = {"luzon", "visayas", "mindanao",
                  "visayan", "mindanaoan", "luzonian"}


# ---------------------------------------------------------------------------
# Loading the blocklist
# ---------------------------------------------------------------------------

def load_blocklist(path: Path) -> list[str]:
    """Load and return the entity list. Order matters: longer phrases first
    so that 'metropolitan manila' is replaced before 'manila' alone."""
    if not path.exists():
        raise FileNotFoundError(f"Place-name blocklist not found: {path}")
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line.lower())
    # De-dup, sort by length descending so longest matches replace first
    entries = sorted(set(entries), key=lambda s: (-len(s), s))
    return entries


def categorize(entity: str) -> str:
    """Assign a category to a blocklist entry."""
    e = entity.lower()
    if e in _REGIONS:
        return "region"
    if e in _METROS:
        return "metropolitan_area"
    if e in _LANDMARKS:
        return "landmark"
    if e in _ISLAND_GROUPS:
        return "island_group"
    # Heuristic: ends with " city" → city; otherwise province; some HUCs
    # don't end with "city" but are well-known cities — handled by hardcoded set.
    KNOWN_CITIES = {
        "manila", "quezon city", "caloocan", "las pinas", "las piñas",
        "makati", "malabon", "mandaluyong", "marikina", "muntinlupa",
        "navotas", "paranaque", "parañaque", "pasay", "pasig", "pateros",
        "san juan", "taguig", "valenzuela", "olongapo", "baguio",
        "dagupan", "lucena", "naga city", "naga", "legazpi", "ligao",
        "tabaco", "iloilo city", "bacolod", "cebu city", "danao", "mandaue",
        "ormoc", "tacloban", "maasin", "calbayog", "catbalogan", "borongan",
        "dapitan", "dipolog", "pagadian", "zamboanga", "cagayan de oro",
        "gingoog", "iligan", "malaybalay", "oroquieta", "ozamiz", "ozamis",
        "tangub", "valencia", "davao city", "digos", "mati", "panabo",
        "samal", "tagum", "general santos", "gensan", "kidapawan", "koronadal",
        "tacurong", "butuan", "bayugan", "bislig", "cabadbaran", "tandag",
        "marawi", "lamitan", "isabela city", "lapu-lapu",
    }
    if e in KNOWN_CITIES or e.endswith(" city"):
        return "city"
    return "province"  # default — most remaining entries are provinces


def build_categories(blocklist: list[str]):
    for e in blocklist:
        ENTITY_CATEGORIES[e] = categorize(e)


# ---------------------------------------------------------------------------
# Replacement engine
# ---------------------------------------------------------------------------

def _make_pseudonym_resolver(seed: int = 1729):
    """Returns a closure that maps real entity → pseudonym, deterministic
    per-run. Pools cycle through their pseudonym pool in order of first
    appearance, with overflow that recycles via numeric suffixes."""
    counters = {
        "city": 0, "province": 0, "region": 0, "landmark": 0,
        "island_group": 0, "metropolitan_area": 0,
    }
    mapping: dict[str, str] = {}
    pools = {
        "city": CITY_POOL,
        "province": PROVINCE_POOL,
        "region": REGION_POOL,
        "island_group": ISLAND_GROUP_POOL,
    }

    def resolve(entity: str, category: str) -> str:
        key = entity.lower()
        if key in mapping:
            return mapping[key]
        if category == "landmark":
            mapping[key] = LANDMARK_LABEL
        elif category == "metropolitan_area":
            # Metro Manila → "Capital Metro Area" (single canonical form)
            mapping[key] = "Capital Metro Area"
        else:
            pool = pools.get(category, CITY_POOL)
            idx = counters[category]
            if idx < len(pool):
                pseudonym = pool[idx]
            else:
                pseudonym = f"{pool[0].rstrip(' WXYZVUTSRQ')} {idx + 1}"
            mapping[key] = pseudonym
            counters[category] += 1
        return mapping[key]

    return resolve, mapping


def _build_pattern(blocklist: list[str]) -> re.Pattern:
    """Compile a single big alternation regex for the entity list. Word-boundary
    anchored; case-insensitive. Multi-word entries handled because they're
    pre-sorted by length descending."""
    parts = [re.escape(e) for e in blocklist]
    pattern = r"(?<![A-Za-z0-9])(" + "|".join(parts) + r")(?![A-Za-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


def replace_in_text(text: str, pattern: re.Pattern, resolver) -> tuple[str, list[tuple[str, str]]]:
    """Run the substitution. Returns (new_text, [(matched_real, pseudonym), ...])."""
    replacements: list[tuple[str, str]] = []

    def _sub(match):
        real = match.group(0)
        category = ENTITY_CATEGORIES.get(real.lower(), "province")
        pseudo = resolver(real, category)
        replacements.append((real, pseudo))
        return pseudo

    new_text = pattern.sub(_sub, text)
    return new_text, replacements


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def pseudonymize_cards(in_path: Path, out_path: Path,
                       blocklist_path: Path, report_path: Path,
                       seed: int = 1729) -> dict:
    cards = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(cards, list):
        raise ValueError(f"{in_path} should be a JSON array of cards; "
                         f"got {type(cards).__name__}")

    blocklist = load_blocklist(blocklist_path)
    build_categories(blocklist)
    pattern = _build_pattern(blocklist)
    resolver, mapping = _make_pseudonym_resolver(seed=seed)

    cards_modified = 0
    total_replacements = 0
    by_category: Counter = Counter()
    top_replaced: Counter = Counter()
    per_card_replacements = defaultdict(list)

    for card in cards:
        text = card.get("text", "")
        if not text:
            continue
        new_text, reps = replace_in_text(text, pattern, resolver)
        if reps:
            cards_modified += 1
            total_replacements += len(reps)
            for real, pseudo in reps:
                top_replaced[real.lower()] += 1
                by_category[ENTITY_CATEGORIES.get(real.lower(), "province")] += 1
                per_card_replacements[card.get("id", "?")].append({
                    "real": real, "pseudo": pseudo
                })
            card["text"] = new_text
            # Audit hook: keep the per-card map alongside the card in metadata
            card.setdefault("metadata", {}).setdefault(
                "place_pseudonyms", []
            ).extend([{"real": r, "pseudo": p} for r, p in reps])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "version": "v2.9.4",
        "input": str(in_path),
        "output": str(out_path),
        "blocklist": str(blocklist_path),
        "blocklist_size": len(blocklist),
        "total_cards": len(cards),
        "cards_modified": cards_modified,
        "total_replacements": total_replacements,
        "by_category": dict(by_category),
        "top_30_replaced": top_replaced.most_common(30),
        "mapping_size": len(mapping),
        "mapping_sample_first10": dict(list(mapping.items())[:10]),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    logger.info("Place-name pseudonymization complete:")
    logger.info("  Cards in           : %d", len(cards))
    logger.info("  Cards modified     : %d", cards_modified)
    logger.info("  Total replacements : %d", total_replacements)
    logger.info("  By category        : %s", dict(by_category))
    logger.info("  Top 5 replaced     : %s", top_replaced.most_common(5))
    return report


def main():
    p = argparse.ArgumentParser(
        description="v2.9.0 — Pseudonymize Philippine geographic entities in card text."
    )
    p.add_argument("--in_file", required=True,
                   help="Input: cards JSON (e.g. generated/cards_pseudo.json)")
    p.add_argument("--out_file", required=True,
                   help="Output: cards JSON with place names pseudonymized")
    p.add_argument("--blocklist", default="templates/places_blocklist.txt",
                   help="Place-name blocklist file")
    p.add_argument("--report_out", default="reports/pseudo_places.json")
    p.add_argument("--seed", type=int, default=1729,
                   help="Deterministic seed for pseudonym assignment")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    pseudonymize_cards(
        in_path=Path(args.in_file),
        out_path=Path(args.out_file),
        blocklist_path=Path(args.blocklist),
        report_path=Path(args.report_out),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

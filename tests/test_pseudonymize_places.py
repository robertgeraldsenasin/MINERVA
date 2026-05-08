"""Unit tests for v2.9.0 scripts/35_pseudonymize_places.py.

Verifies:
  - All 4 leaked names from the v2.8.7 audit get replaced
  - Mapping is deterministic (same real entity → same pseudonym across run)
  - Categories are correct (city / province / region / landmark)
  - Long entries replace before short (Metropolitan Manila before Manila)
  - Word-boundary protection prevents partial-word matches
  - Per-card audit metadata is attached

Run:
    python -m pytest tests/test_pseudonymize_places.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "35_pseudonymize_places.py"
BLOCKLIST = REPO_ROOT / "templates" / "places_blocklist.txt"


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("pseudo35", str(SCRIPT_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pseudo35"] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


# ----------------------------------------------------------------------
# Blocklist loading
# ----------------------------------------------------------------------

class TestBlocklistLoad:
    def test_loads_and_sorts_by_length_desc(self):
        bl = m.load_blocklist(BLOCKLIST)
        assert len(bl) > 100  # comprehensive
        # First entry should be one of the longest
        assert len(bl[0]) >= len(bl[-1])

    def test_excludes_comments_and_blanks(self):
        bl = m.load_blocklist(BLOCKLIST)
        for entry in bl:
            assert not entry.startswith("#")
            assert entry.strip() != ""


# ----------------------------------------------------------------------
# The 4 audit-reported leaks
# ----------------------------------------------------------------------

class TestAuditLeakedNames:
    """The v2.8.7 audit found these 4 names leaking. They MUST be caught."""

    def setup_method(self):
        self.bl = m.load_blocklist(BLOCKLIST)
        m.build_categories(self.bl)
        self.pattern = m._build_pattern(self.bl)
        self.resolver, self.mapping = m._make_pseudonym_resolver()

    def test_lapulapu_city_replaced(self):
        text = "Sa Lapu-Lapu City, isinagawa ang programa."
        out, reps = m.replace_in_text(text, self.pattern, self.resolver)
        assert "Lapu-Lapu City" not in out
        assert len(reps) == 1
        assert reps[0][0] == "Lapu-Lapu City"
        assert reps[0][1].startswith("City")

    def test_baguio_city_replaced(self):
        text = "Mga turista sa Baguio City."
        out, reps = m.replace_in_text(text, self.pattern, self.resolver)
        assert "Baguio City" not in out
        assert len(reps) == 1

    def test_pasay_city_replaced(self):
        text = "Sa Pasay City, may bagong programa."
        out, reps = m.replace_in_text(text, self.pattern, self.resolver)
        assert "Pasay City" not in out
        assert len(reps) == 1

    def test_metropolitan_manila_replaced_as_metro(self):
        text = "Sa buong Metropolitan Manila ay napakaraming residente."
        out, reps = m.replace_in_text(text, self.pattern, self.resolver)
        assert "Metropolitan Manila" not in out
        # Should map to "Capital Metro Area" (single canonical metro pseudonym)
        assert reps[0][1] == "Capital Metro Area"


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------

class TestDeterminism:
    def test_same_entity_same_pseudonym_within_run(self):
        bl = m.load_blocklist(BLOCKLIST)
        m.build_categories(bl)
        pattern = m._build_pattern(bl)
        resolver, _ = m._make_pseudonym_resolver()

        text1 = "Sa Cebu City, may konsentrasyon ng tao."
        text2 = "Bumalik ako sa Cebu City kahapon."
        out1, _ = m.replace_in_text(text1, pattern, resolver)
        out2, _ = m.replace_in_text(text2, pattern, resolver)

        # The pseudonym for Cebu City must be the same across both texts
        # Extract: the first word after substitution should match
        cebu_pseudonym_1 = out1.split("Sa ")[1].split(",")[0]
        cebu_pseudonym_2 = out2.split("sa ")[1].split(" kahapon")[0]
        assert cebu_pseudonym_1 == cebu_pseudonym_2

    def test_different_entities_different_pseudonyms(self):
        bl = m.load_blocklist(BLOCKLIST)
        m.build_categories(bl)
        pattern = m._build_pattern(bl)
        resolver, _ = m._make_pseudonym_resolver()

        text = "Manggagaling sa Cebu City papuntang Davao City."
        out, reps = m.replace_in_text(text, pattern, resolver)
        # Two cities, two distinct pseudonyms
        cebu_pseudo = next(p for r, p in reps if r == "Cebu City")
        davao_pseudo = next(p for r, p in reps if r == "Davao City")
        assert cebu_pseudo != davao_pseudo


# ----------------------------------------------------------------------
# Categorization
# ----------------------------------------------------------------------

class TestCategorization:
    def test_known_cities_categorized_as_city(self):
        for c in ["manila", "quezon city", "cebu city", "davao city",
                  "baguio", "lapu-lapu city"]:
            assert m.categorize(c) == "city"

    def test_regions_categorized_as_region(self):
        assert m.categorize("calabarzon") == "region"
        assert m.categorize("national capital region") == "region"
        assert m.categorize("bangsamoro") == "region"

    def test_landmarks_categorized_as_landmark(self):
        for ld in ["malacanang", "edsa", "manila bay", "fort bonifacio"]:
            assert m.categorize(ld) == "landmark"

    def test_island_groups_categorized(self):
        for ig in ["luzon", "visayas", "mindanao"]:
            assert m.categorize(ig) == "island_group"

    def test_metro_categorized(self):
        assert m.categorize("metropolitan manila") == "metropolitan_area"


# ----------------------------------------------------------------------
# Word-boundary correctness
# ----------------------------------------------------------------------

class TestWordBoundaries:
    def test_does_not_match_substring_inside_word(self):
        bl = m.load_blocklist(BLOCKLIST)
        m.build_categories(bl)
        pattern = m._build_pattern(bl)
        resolver, _ = m._make_pseudonym_resolver()

        # "Manila" is a city; "Manilakaw" (made up) should NOT be replaced
        text = "Ang Manilakaw ay isang custom term."
        out, reps = m.replace_in_text(text, pattern, resolver)
        assert "Manilakaw" in out
        assert len(reps) == 0

    def test_long_phrase_replaces_before_short(self):
        """Metropolitan Manila must be matched as a unit, not as 'Manila'."""
        bl = m.load_blocklist(BLOCKLIST)
        m.build_categories(bl)
        pattern = m._build_pattern(bl)
        resolver, _ = m._make_pseudonym_resolver()

        text = "Sa Metropolitan Manila."
        out, reps = m.replace_in_text(text, pattern, resolver)
        # Should be ONE replacement (the whole "Metropolitan Manila" phrase),
        # not two (one for "Metropolitan Manila" and another for "Manila")
        assert len(reps) == 1
        assert reps[0][0] == "Metropolitan Manila"


# ----------------------------------------------------------------------
# End-to-end: pseudonymize_cards driver
# ----------------------------------------------------------------------

class TestEndToEnd:
    def test_processes_full_card_array_and_writes_report(self, tmp_path):
        cards = [
            {"id": "test_001", "text": "Sa Lapu-Lapu City may bagong proyekto."},
            {"id": "test_002", "text": "Si Candidate A ay nagsalita sa Baguio City."},
            {"id": "test_003", "text": "Walang lugar na binanggit dito."},  # no place
        ]
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "out.json"
        report_path = tmp_path / "report.json"
        in_path.write_text(json.dumps(cards), encoding="utf-8")

        report = m.pseudonymize_cards(
            in_path=in_path, out_path=out_path,
            blocklist_path=BLOCKLIST, report_path=report_path,
        )

        assert report["total_cards"] == 3
        assert report["cards_modified"] == 2  # cards 1 and 2
        assert report["total_replacements"] == 2
        assert report["by_category"].get("city", 0) == 2

        out_cards = json.loads(out_path.read_text(encoding="utf-8"))
        assert "Lapu-Lapu City" not in out_cards[0]["text"]
        assert "Baguio City" not in out_cards[1]["text"]
        assert out_cards[2]["text"] == "Walang lugar na binanggit dito."

        # Per-card audit metadata
        assert "place_pseudonyms" in out_cards[0]["metadata"]
        assert len(out_cards[0]["metadata"]["place_pseudonyms"]) == 1


# ----------------------------------------------------------------------
# Coverage: blocklist is comprehensive enough
# ----------------------------------------------------------------------

class TestCoverage:
    def test_all_17_regions_in_blocklist(self):
        bl = m.load_blocklist(BLOCKLIST)
        regions = [
            "ilocos region", "cagayan valley", "central luzon", "calabarzon",
            "mimaropa", "bicol region", "western visayas", "central visayas",
            "eastern visayas", "zamboanga peninsula", "northern mindanao",
            "davao region", "soccsksargen", "caraga", "bangsamoro",
            "cordillera administrative region", "national capital region",
        ]
        bl_set = set(bl)
        for r in regions:
            assert r in bl_set, f"Missing region: {r}"

    def test_top_10_metro_manila_cities_present(self):
        bl_set = set(m.load_blocklist(BLOCKLIST))
        for c in ["manila", "quezon city", "makati", "pasig", "taguig",
                  "pasay", "marikina", "muntinlupa", "valenzuela", "caloocan"]:
            assert c in bl_set, f"Missing MM city: {c}"

    def test_blocklist_covers_audit_leaks(self):
        bl_set = set(m.load_blocklist(BLOCKLIST))
        # The exact 4 leaks the v2.8.7 audit found
        assert "lapu-lapu city" in bl_set
        assert "baguio city" in bl_set
        assert "pasay city" in bl_set
        assert "metropolitan manila" in bl_set

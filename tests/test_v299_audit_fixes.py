"""Regression tests for v2.9.9: allowlist closure (8 generic edge cases + tondo to places_blocklist)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    import importlib.util
    for mod in list(sys.modules):
        if name in mod or "minerva" in mod:
            sys.modules.pop(mod, None)
    spec = importlib.util.spec_from_file_location(
        name, str(REPO_ROOT / "scripts" / path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestV299AllowlistFinalClosure:
    """v2.9.8 run zip surfaced 11 generic unknown_name rejections; v2.9.9 covers all."""

    def test_tagalog_role_titles_allowed(self):
        m = _load_module("m33_v299", "33_strict_name_allowlist.py")
        allowed = m._ALLOWED_ORGANIZATIONS
        for term in ["presidente", "presidential",
                     "education sec", "education secretary"]:
            assert term in allowed, f"{term!r} should be allowed in v2.9.9"

    def test_pnp_unit_names_allowed(self):
        m = _load_module("m33_v299", "33_strict_name_allowlist.py")
        allowed = m._ALLOWED_ORGANIZATIONS
        for term in ["intelligence group", "pnp intelligence group",
                     "investigation group"]:
            assert term in allowed, f"{term!r} should be allowed in v2.9.9"

    def test_news_program_titles_allowed(self):
        """Filipino news program names like 'Unang Balita', '24 Oras' are
        press citations, not 'candidate names'. Same status as 'rappler'
        and 'inquirer' which were already in the allowlist."""
        m = _load_module("m33_v299", "33_strict_name_allowlist.py")
        allowed = m._ALLOWED_ORGANIZATIONS
        for term in ["unang balita", "24 oras", "tv patrol"]:
            assert term in allowed, f"{term!r} should be allowed in v2.9.9"

    def test_filipino_colloquial_terms_allowed(self):
        """'Isang Pinay', 'Pinoy' — gendered/cultural generic terms,
        not naming any specific individual."""
        m = _load_module("m33_v299", "33_strict_name_allowlist.py")
        allowed = m._ALLOWED_ORGANIZATIONS
        for term in ["isang pinay", "pinay", "pinoy"]:
            assert term in allowed, f"{term!r} should be allowed in v2.9.9"

    def test_foreign_nationalities_allowed_when_adjective(self):
        """'Japanese vlogger', 'Indonesian official' — adjectival use, person
        still pseudonymized to Candidate A/B/C. Like 'Japanese' in 'Japanese
        car'; descriptive, not naming."""
        m = _load_module("m33_v299", "33_strict_name_allowlist.py")
        allowed = m._ALLOWED_ORGANIZATIONS
        for term in ["japanese", "indonesia", "indonesian", "korean",
                     "the japan", "the korea"]:
            assert term in allowed, f"{term!r} should be allowed in v2.9.9"


class TestV299PlacesBlocklistTondo:
    """Tondo is a famous Manila district. The place pseudonymizer should
    catch it (it gets pseudonymized to e.g. 'District K' before reaching
    the strict allowlist)."""

    def test_tondo_in_places_blocklist(self):
        blocklist_path = REPO_ROOT / "templates" / "places_blocklist.txt"
        content = blocklist_path.read_text(encoding="utf-8").lower()
        # Verify "tondo" appears as its own line (not just substring of another word)
        lines = [line.strip() for line in content.splitlines()
                 if line.strip() and not line.strip().startswith("#")]
        assert "tondo" in lines, (
            "'tondo' should be in places_blocklist.txt so the place "
            "pseudonymizer (script 35) catches it. Otherwise it leaks "
            "all the way to strict allowlist and gets flagged."
        )


class TestV299RealPoliticalNamesStillBlocked:
    """Defense-in-depth: the 3 blocklist-matched rejections from v2.9.8
    (Aquino, Kiko, Villar) MUST remain blocked. These are real Filipino
    political dynasty surnames."""

    def test_aquino_still_blocked(self):
        """Aquino → real political dynasty (Ninoy, Cory, Noynoy)."""
        m = _load_module("m33_v299", "33_strict_name_allowlist.py")
        # Aquino should NOT be in allowed; it should be in fallback blocklist
        assert "aquino" not in m._ALLOWED_ORGANIZATIONS, (
            "'aquino' must NOT be allowed — it's a real political dynasty name"
        )
        assert "aquino" in m._FALLBACK_BLOCKLIST, (
            "'aquino' must remain in _FALLBACK_BLOCKLIST"
        )

    def test_villar_still_blocked(self):
        m = _load_module("m33_v299", "33_strict_name_allowlist.py")
        assert "villar" not in m._ALLOWED_ORGANIZATIONS
        assert "villar" in m._FALLBACK_BLOCKLIST

    def test_marcos_still_blocked(self):
        m = _load_module("m33_v299", "33_strict_name_allowlist.py")
        assert "marcos" not in m._ALLOWED_ORGANIZATIONS
        assert "marcos" in m._FALLBACK_BLOCKLIST

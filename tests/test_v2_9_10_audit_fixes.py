"""Regression tests for v2.9.10: optional polish — 5 generic agency/calendar terms."""
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


class TestV2_9_10AgencyTermsAllowed:
    """Generic agency / unit names that GPT-2 used descriptively."""

    def test_election_agencies_allowed(self):
        m = _load_module("m33_v2910", "33_strict_name_allowlist.py")
        allowed = m._ALLOWED_ORGANIZATIONS
        for term in ["election board", "election commission"]:
            assert term in allowed, f"{term!r} should be allowed in v2.9.10"

    def test_law_enforcement_agencies_allowed(self):
        m = _load_module("m33_v2910", "33_strict_name_allowlist.py")
        allowed = m._ALLOWED_ORGANIZATIONS
        for term in ["national bureau", "national bureau of investigation",
                     "cybercrime division", "cybercrime unit"]:
            assert term in allowed, f"{term!r} should be allowed in v2.9.10"

    def test_calendar_terms_allowed(self):
        """Calendar events like 'Valentine's Day' are not personal names."""
        m = _load_module("m33_v2910", "33_strict_name_allowlist.py")
        allowed = m._ALLOWED_ORGANIZATIONS
        for term in ["valentine's day", "valentines day", "christmas", "holy week"]:
            assert term in allowed, f"{term!r} should be allowed in v2.9.10"


class TestV2_9_10PacquiaoStillBlocked:
    """Defense-in-depth: Manny Pacquiao is a real senator. MUST stay blocked."""

    def test_pacquiao_still_blocked(self):
        m = _load_module("m33_v2910", "33_strict_name_allowlist.py")
        assert "pacquiao" not in m._ALLOWED_ORGANIZATIONS, (
            "Pacquiao is a real PH senator — must NOT be in allowlist"
        )
        assert "pacquiao" in m._FALLBACK_BLOCKLIST, (
            "Pacquiao must remain in _FALLBACK_BLOCKLIST"
        )

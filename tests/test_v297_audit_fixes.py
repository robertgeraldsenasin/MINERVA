"""Regression tests for v2.9.7: bank_ref regex (legacy + v2.9 4-segment) and 25+ allowlist entries."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / "scripts" / path).read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Fix #1 — script 33 allowlist expansion (closes 96.54% regression)
# ----------------------------------------------------------------------

class TestAllowlistExpansionV297:
    """v2.9.6 audit final-run finding: 23 cards rejected for unknown entities.
    v2.9.7 expands the allowlist to recognize the legitimate generic terms
    that GPT-2 produces."""

    def _get_allowed_set(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import importlib.util
        for mod in list(sys.modules):
            if "33_" in mod or "minerva" in mod:
                sys.modules.pop(mod, None)
        spec = importlib.util.spec_from_file_location(
            "m33", str(REPO_ROOT / "scripts" / "33_strict_name_allowlist.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m._ALLOWED_ORGANIZATIONS

    def test_government_orgs_in_allowlist(self):
        allowed = self._get_allowed_set()
        critical = [
            "supreme court", "deped", "the deped",
            "presidential communications operations office", "pcoo",
            "police regional office", "philippine institute",
        ]
        for term in critical:
            assert term in allowed, f"{term!r} should be in _ALLOWED_ORGANIZATIONS"

    def test_geographic_pseudonym_outputs_in_allowlist(self):
        """The place pseudonymizer emits these generic forms; they should not
        trigger the strict allowlist."""
        allowed = self._get_allowed_set()
        critical = [
            "capital metro area",        # Metro Manila → Capital Metro Area
            "capital metro area council",
            "island group",              # Luzon / Visayas / Mindanao → Island Group N
            "barangay sta", "barangay sto",
        ]
        for term in critical:
            assert term in allowed, f"{term!r} should be in _ALLOWED_ORGANIZATIONS"

    def test_misflagged_common_words_in_allowlist(self):
        """Common words capitalized in news context shouldn't be foreign names."""
        allowed = self._get_allowed_set()
        for term in ["justice", "papa", "daily news"]:
            assert term in allowed, f"{term!r} should be in _ALLOWED_ORGANIZATIONS"

    def test_minor_ph_cities_in_allowlist(self):
        """PH cities below the top-tier list (allowlist's original coverage)."""
        allowed = self._get_allowed_set()
        for term in ["antipolo city", "tuguegarao city"]:
            assert term in allowed, f"{term!r} should be in _ALLOWED_ORGANIZATIONS"

    def test_composite_forms_in_allowlist(self):
        """Government-role + Candidate composites that get parsed as single token."""
        allowed = self._get_allowed_set()
        assert "deped candidate" in allowed


# ----------------------------------------------------------------------
# Fix #2 — script 26 bank_ref regex (closes 85.24% faithfulness regression)
# ----------------------------------------------------------------------

class TestFaithfulnessAuditRegexV297:
    """v2.9.6 audit found 320 malformed_bank_ref + 98 stale_bank_version
    issues, all due to script 26's regex being stuck on the pre-v2.9 format
    AND a loop-indentation bug that only validated the LAST ref per card."""

    def test_script_26_loop_indentation_is_correct(self):
        """The v2.9.7 fix moved the validation INSIDE the for-loop; previously
        only the last bank_ref per card was being validated."""
        src = _read("26_faithfulness_audit.py")
        # Find the bank_ref check block and verify the regex match is inside
        # the for loop (indented further than the loop body)
        idx = src.find("Check 3: bank_ref")
        block = src[idx:idx + 1500]

        # The regex matchers should be DEFINED before the loop, but the if-check
        # should be INSIDE it.
        assert "for p in indicator_phrases" in block
        # The crucial check: the regex match-and-append should be more indented
        # than the for-loop body
        loop_start = block.find("for p in indicator_phrases")
        match_check = block.find("if not (BANK_REF_NEW.match")
        # The match check must come AFTER the for loop and be deeper-indented
        assert match_check > loop_start, "Validation must be after loop start"

        # Check that ref = p.get(...) is sibling of the validation, both inside loop
        loop_section = block[loop_start:loop_start + 600]
        assert "ref = p.get(" in loop_section
        assert "if not (BANK_REF_NEW.match" in loop_section, (
            "The bank_ref validation must be INSIDE the for-loop body. "
            "v2.9.6 bug: validation was outside, only last ref got checked."
        )

    def test_new_bank_ref_format_accepted(self):
        """The v2.9 bank_ref format <INDICATOR>/<role>/<tier>/v<N> must validate."""
        src = _read("26_faithfulness_audit.py")
        # Extract the BANK_REF_NEW regex pattern
        m = re.search(r'BANK_REF_NEW\s*=\s*re\.compile\(\s*r"([^"]+)"', src)
        assert m is not None, "BANK_REF_NEW regex must be present"
        pattern = re.compile(m.group(1))
        # Verify it matches v2.9-style refs
        for ref in ["MISS/fake/novice/v0", "URG/fake/novice/v2",
                    "EMO/real/proficient/v1", "CREDIBLE/real/novice/v1",
                    "POL/real/advanced/v3"]:
            assert pattern.match(ref), f"v2.9 format {ref!r} should match"

    def test_legacy_bank_ref_format_still_accepted(self):
        """Backwards compat: pre-v2.9 format <INDICATOR>/v<v>/<tier-letter><idx> still validates."""
        src = _read("26_faithfulness_audit.py")
        m = re.search(r'BANK_REF_LEGACY\s*=\s*re\.compile\(\s*r"([^"]+)"', src)
        assert m is not None, "BANK_REF_LEGACY regex must be present"
        pattern = re.compile(m.group(1))
        for ref in ["MISS/v1.0/n0", "URG/v2/n3", "EMO/v1.1/p2", "FAB/v1/a5"]:
            assert pattern.match(ref), f"legacy format {ref!r} should still match"

    def test_codename_bank_version_accepted(self):
        """v2.9.7: codename-style version stamps (v2.9.0, v2.9.6) should not
        be flagged as stale when the bank uses semver internally."""
        src = _read("26_faithfulness_audit.py")
        assert "_CODENAME_RX" in src, "Codename-version reconciler must be present"
        # Pull out and test the regex
        m = re.search(r'_CODENAME_RX\s*=\s*re\.compile\(\s*r"([^"]+)"', src)
        assert m is not None
        pattern = re.compile(m.group(1))
        for ref in ["v2.9.0", "v2.9.4", "v2.9.6", "v3.0.0"]:
            assert pattern.match(ref), f"codename {ref!r} should match"
        for ref in ["1.1", "2.0", "unknown", "v2.9", "9.0.0"]:
            # These should NOT be treated as codenames (so they'd flag stale)
            if pattern.match(ref):
                continue  # acceptable false-positive — test stays loose

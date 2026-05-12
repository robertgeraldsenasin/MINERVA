"""Regression tests for v2.9.5 audit-driven code fixes.

These lock in:
  - GPT-2 training default seed is non-42 (Picard 2021 fix)
  - Scripts 35/37 stamp v2.9.4 (was v2.9.0)
  - Script 21 emits schema-invalid categorization (was just a count)
  - Script 32 emits the tautology caveat (was silent about it)

Run:
    python -m pytest tests/test_v295_audit_fixes.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _read(path: str) -> str:
    return (SCRIPTS_DIR / path).read_text(encoding="utf-8")


# ----------------------------------------------------------------------
# Fix #1 — GPT-2 training seed is non-42 (Picard 2021)
# ----------------------------------------------------------------------

class TestGPT2TrainingSeed:
    """GPT-2 fine-tune default seed must not be 42.
    Audit ref: v2.9.4 final-run audit, MEDIUM #3."""

    def test_default_seed_is_not_42(self):
        src = _read("11b_train_gpt2_neurosymbolic.py")
        # Look for --seed default
        m = re.search(r'add_argument\("--seed",\s*type=int,\s*default=(\d+)\)', src)
        assert m is not None, "Couldn't find --seed argparse default"
        seed = int(m.group(1))
        assert seed != 42, (
            f"GPT-2 default seed is {seed}; should be non-42 per "
            "Picard (2021) cherry-picking critique. Recommended: 1729 "
            "(Hardy-Ramanujan) for codebase consistency."
        )

    def test_default_seed_is_1729(self):
        """Specific value check — keeps the codebase aligned with
        scripts/minerva_candidates.py:305 which also uses 1729."""
        src = _read("11b_train_gpt2_neurosymbolic.py")
        m = re.search(r'add_argument\("--seed",\s*type=int,\s*default=(\d+)\)', src)
        assert m and int(m.group(1)) == 1729


# ----------------------------------------------------------------------
# Fix #2 + 3 — Version stamps in scripts 35 + 37
# ----------------------------------------------------------------------

class TestVersionStamps:
    """v2.9.4 audit final-run report's LOW #6: scripts 35, 37 still stamp
    v2.9.0 while corpus + training + generation all stamp v2.9.4."""

    def test_script_35_stamps_v294(self):
        src = _read("35_pseudonymize_places.py")
        assert '"version": "v2.9.0"' not in src, (
            "Script 35 still stamps v2.9.0 — should be v2.9.4 for consistency"
        )
        assert '"version": "v2.9.4"' in src

    def test_script_37_stamps_v294(self):
        src = _read("37_holdout_detector_eval.py")
        assert '"version": "v2.9.0"' not in src, (
            "Script 37 still stamps v2.9.0 — should be v2.9.4 for consistency"
        )
        assert '"version": "v2.9.4"' in src


# ----------------------------------------------------------------------
# Fix #4 — Schema-invalid categorization in script 21
# ----------------------------------------------------------------------

class TestSchemaInvalidDiagnostic:
    """v2.9.4 audit MEDIUM #4: script 21 drops ~31.5% of cards as
    schema-invalid but doesn't surface WHY. v2.9.5 categorizes the
    failure modes."""

    def test_script_21_categorizes_failures(self):
        src = _read("21_balance_unity_cards.py")
        assert "invalid_by_reason" in src, (
            "Script 21 should emit categorized schema-invalid reasons"
        )

    def test_script_21_emits_examples(self):
        src = _read("21_balance_unity_cards.py")
        assert "invalid_examples" in src, (
            "Script 21 should keep first-10 schema-invalid examples"
        )

    def test_script_21_report_has_new_fields(self):
        """The report dict in script 21 should now include the new fields."""
        src = _read("21_balance_unity_cards.py")
        assert '"schema_invalid_by_reason"' in src
        assert '"schema_invalid_examples_first10"' in src

    def test_script_21_categorization_has_known_buckets(self):
        """Specific categories the v2.9.5 logic must recognize."""
        src = _read("21_balance_unity_cards.py")
        expected_categories = [
            "missing_required_field",
            "wrong_field_type",
            "invalid_candidate_code",
            "invalid_verdict",
            "invalid_indicator",
            "other",
        ]
        for cat in expected_categories:
            assert f'"{cat}"' in src, (
                f"Schema-invalid category {cat!r} should be in script 21"
            )


# ----------------------------------------------------------------------
# Fix #5 — det.json tautology caveat in script 32
# ----------------------------------------------------------------------

class TestDetectorValidationCaveat:
    """v2.9.4 audit MEDIUM #5: det.json shows 100% across all four detectors
    because the pool was curated to detector consensus. v2.9.5 surfaces
    this caveat explicitly in the report dict."""

    def test_script_32_has_interpretation_field(self):
        src = _read("32_validate_detectors_on_templates.py")
        assert '"interpretation"' in src, (
            "Script 32's report should include an 'interpretation' field"
        )

    def test_script_32_explicitly_calls_metric_internal(self):
        src = _read("32_validate_detectors_on_templates.py")
        assert '"metric_kind": "internal_consensus"' in src

    def test_script_32_redirects_to_holdout(self):
        """The caveat should point readers to the real generalization metric."""
        src = _read("32_validate_detectors_on_templates.py")
        assert "holdout_detector_eval.json" in src, (
            "Script 32's interpretation should redirect to script 37's "
            "holdout eval for the real generalization F1."
        )

    def test_script_32_does_not_claim_generalization(self):
        """The caveat text must use disambiguating language."""
        src = _read("32_validate_detectors_on_templates.py")
        # Look for the interpretation string
        m = re.search(r'"interpretation":\s*\(\s*(["\'].*?["\'])\s*"', src, re.DOTALL)
        # As long as the file mentions both these things together, we're good
        assert "INTERNAL-CONSISTENCY" in src or "internal-consistency" in src.lower()
        assert "NOT a generalization" in src or "not a generalization" in src.lower()

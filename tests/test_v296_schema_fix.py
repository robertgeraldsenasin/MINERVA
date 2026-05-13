"""Regression test for v2.9.6 audit-driven schema fix.

The v2.9.5 final-run audit revealed that 523/1423 (37%) of post-merge cards
drop at script 21's schema validation with reason "invalid_indicator". The
v2.9.5 schema-invalid-by-reason diagnostic surfaced the categorization, but
the underlying cause was actually `extra_forbidden` errors on
`IndicatorPhrase.phrase_en` and `IndicatorPhrase.verifier_action` fields.

The response_bank_v2.json supplies these fields alongside `phrase_tl`, and
script 29 passes them through unchanged. The v2.9.6 fix adds them to the
schema as optional fields.

Confirmed simulation result on the v2.9.5 run zip's
generated/template_plus_gpt2_cards.json:
  Before fix: 900/1423 valid (63.3%) — 523 dropped
  After fix:  1423/1423 valid (100.0%) — 0 dropped
  Recovered ~37% of the pool, ALL of which were GPT-2 cards.

Run:
    python -m pytest tests/test_v296_schema_fix.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from minerva_schemas import IndicatorPhrase, UnityCard


class TestIndicatorPhraseExtraFields:
    """IndicatorPhrase must accept phrase_en and verifier_action (from
    response_bank_v2.json). v2.9.6 fix."""

    def test_phrase_en_field_accepted(self):
        """phrase_en is an optional English-language fallback for the
        Unity UI; was rejected by extra='forbid' before v2.9.6."""
        ip = IndicatorPhrase(
            indicator="EMO",
            phrase="Loaded na salita imbis na ebidensya.",
            bank_ref="EMO/fake/novice/v0",
            phrase_en="Loaded language instead of evidence.",
        )
        assert ip.phrase_en == "Loaded language instead of evidence."

    def test_verifier_action_field_accepted(self):
        """verifier_action is the SIFT-style verification suggestion."""
        ip = IndicatorPhrase(
            indicator="EMO",
            phrase="Loaded na salita imbis na ebidensya.",
            bank_ref="EMO/fake/novice/v0",
            verifier_action="Tanungin: 'May ebidensya ba o emosyon lang?'",
        )
        assert ip.verifier_action.startswith("Tanungin")

    def test_both_extra_fields_simultaneously(self):
        """Real-world response_bank_v2.json entries have both fields."""
        ip = IndicatorPhrase(
            indicator="EMO",
            phrase="Loaded na salita.",
            bank_ref="EMO/fake/novice/v0",
            phrase_en="Loaded language.",
            verifier_action="Tanungin: May ebidensya ba?",
        )
        assert ip.phrase_en is not None
        assert ip.verifier_action is not None

    def test_omitting_extras_still_works(self):
        """Backwards compatibility: cards without these fields still validate."""
        ip = IndicatorPhrase(
            indicator="EMO",
            phrase="Salitang gumigising sa galit.",
            bank_ref="EMO/fake/novice/v0",
        )
        assert ip.phrase_en is None
        assert ip.verifier_action is None

    def test_other_extra_fields_still_forbidden(self):
        """Defense-in-depth: we added two specific fields, not turned off
        extra='forbid'. Random extra fields should still be rejected."""
        import pydantic
        try:
            IndicatorPhrase(
                indicator="EMO",
                phrase="phrase text long enough",
                bank_ref="EMO/fake/novice/v0",
                completely_random_field="should be rejected",
            )
            assert False, "Random extra fields should be rejected"
        except pydantic.ValidationError as e:
            assert "extra_forbidden" in str(e) or "Extra inputs" in str(e)


class TestScript21CategorizationRecognizesExtraForbidden:
    """v2.9.6: script 21's invalid_by_reason categorization must
    recognize extra_forbidden as its own category, not lump it under
    other or invalid_indicator."""

    def test_script_21_has_extra_forbidden_category(self):
        src = (REPO_ROOT / "scripts" / "21_balance_unity_cards.py").read_text()
        assert '"extra_forbidden_field"' in src, (
            "Script 21 should recognize extra_forbidden as its own category "
            "to avoid the v2.9.5 misclassification where 523 cards were "
            "labeled invalid_indicator when they were actually extra_forbidden"
        )

    def test_script_21_checks_extra_forbidden_first(self):
        """extra_forbidden should be checked before the generic 'indicator'
        substring check, since the error message also contains the word
        'indicator' when the extra field is on IndicatorPhrase."""
        src = (REPO_ROOT / "scripts" / "21_balance_unity_cards.py").read_text()
        # Find the categorization logic
        idx_extra = src.find('"extra_forbidden_field"')
        idx_indicator = src.find('"invalid_indicator"')
        assert idx_extra < idx_indicator, (
            "extra_forbidden check must come BEFORE the generic indicator check"
        )

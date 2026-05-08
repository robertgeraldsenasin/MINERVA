"""Unit tests for the v2.9.0 response bank (templates/response_bank_v2.json).

Verifies:
  - Bank exists and parses as valid JSON with the expected schema.
  - Coverage: every (indicator × role × tier) combination has ≥1 phrase variant.
  - Diversity: total unique phrases ≥30% of total entries (audit recommendation).
  - Quantity: ≥200 phrase entries (audit recommendation).
  - Each phrase has the required fields (phrase_tl, phrase_en, verifier_action).
  - Bank integrates correctly with scripts/29_merge_gpt2_into_pool.py
    (import + key lookup work end-to-end).

Run:
    python -m pytest tests/test_response_bank.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BANK_PATH = REPO_ROOT / "templates" / "response_bank_v2.json"


@pytest.fixture
def bank():
    if not BANK_PATH.exists():
        pytest.skip(f"Response bank not present: {BANK_PATH}")
    return json.loads(BANK_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------

class TestSchema:
    def test_top_level_keys(self, bank):
        assert "version" in bank
        assert "phrases" in bank
        assert "indicators" in bank

    def test_indicators_have_required_fields(self, bank):
        for code, meta in bank["indicators"].items():
            assert "label" in meta, f"indicator {code} missing label"
            assert "description" in meta, f"indicator {code} missing description"
            assert "default_sift_move" in meta, f"indicator {code} missing default_sift_move"
            assert meta["default_sift_move"] in ("STOP", "TRACE")

    def test_phrase_entries_have_required_fields(self, bank):
        for key, variants in bank["phrases"].items():
            for v in variants:
                assert "phrase_tl" in v, f"key {key}: missing phrase_tl"
                assert "phrase_en" in v, f"key {key}: missing phrase_en"
                assert "verifier_action" in v, f"key {key}: missing verifier_action"
                assert v["phrase_tl"], f"key {key}: empty phrase_tl"


# ----------------------------------------------------------------------
# Quantity
# ----------------------------------------------------------------------

class TestQuantity:
    def test_at_least_200_entries(self, bank):
        total = sum(len(v) for v in bank["phrases"].values())
        assert total >= 200, (
            f"Bank has {total} phrase entries; audit recommendation is ≥200. "
            f"Add more variants to push pool diversity above 30%."
        )

    def test_minimum_keys(self, bank):
        # 12 indicators × {fake, real} × 3 tiers = 72 keys minimum
        # CREDIBLE only has /real/ → +3
        # Total expected: 75
        assert len(bank["phrases"]) >= 72


# ----------------------------------------------------------------------
# Diversity (audit recommendation: ≥30% unique)
# ----------------------------------------------------------------------

class TestDiversity:
    def test_unique_tl_phrase_ratio(self, bank):
        all_phrases = []
        for variants in bank["phrases"].values():
            for v in variants:
                all_phrases.append(v["phrase_tl"])
        unique = set(all_phrases)
        ratio = len(unique) / len(all_phrases)
        assert ratio >= 0.30, (
            f"Unique TL phrase ratio {ratio:.1%} < 30% target. "
            f"Bank has {len(all_phrases)} entries but only "
            f"{len(unique)} unique phrases."
        )

    def test_each_key_has_multiple_variants(self, bank):
        """Each (indicator × role × tier) key should have ≥3 variants so
        the rotation in script 29 produces different phrases."""
        for key, variants in bank["phrases"].items():
            assert len(variants) >= 3, (
                f"Key '{key}' has only {len(variants)} variant(s); ≥3 required."
            )


# ----------------------------------------------------------------------
# Coverage: every indicator × role × tier we expect to use
# ----------------------------------------------------------------------

class TestCoverage:
    def test_covers_all_paper_indicators(self, bank):
        """The 12 indicators referenced in BATB §3.5 + CREDIBLE."""
        expected = {"EMO", "URG", "ANON", "MISS", "FAB", "POL",
                    "CONS", "DISC", "IMP", "REV", "ENDO", "RECF",
                    "CREDIBLE"}
        covered = set(bank["indicators"].keys())
        assert expected.issubset(covered), f"Missing: {expected - covered}"

    def test_each_misinformation_indicator_has_fake_and_real_keys(self, bank):
        non_credible = [c for c in bank["indicators"]
                        if c != "CREDIBLE"]
        keys = set(bank["phrases"].keys())
        for code in non_credible:
            for tier in ("novice", "proficient", "advanced"):
                assert f"{code}/fake/{tier}" in keys, f"Missing {code}/fake/{tier}"
                assert f"{code}/real/{tier}" in keys, f"Missing {code}/real/{tier}"

    def test_credible_only_has_real_keys(self, bank):
        keys = set(bank["phrases"].keys())
        for tier in ("novice", "proficient", "advanced"):
            assert f"CREDIBLE/real/{tier}" in keys
            assert f"CREDIBLE/fake/{tier}" not in keys


# ----------------------------------------------------------------------
# Integration with script 29
# ----------------------------------------------------------------------

class TestScript29Integration:
    def test_merge_module_loads_bank(self, monkeypatch):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "merge29", str(REPO_ROOT / "scripts" / "29_merge_gpt2_into_pool.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Force the path so the loader picks up our bank
        mod._RESPONSE_BANK = None
        bank = mod._load_response_bank(str(BANK_PATH))
        assert bank["version"] != "stub"
        assert "phrases" in bank
        assert len(bank["phrases"]) >= 72

    def test_indicator_coverage_filter_works(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "merge29_b", str(REPO_ROOT / "scripts" / "29_merge_gpt2_into_pool.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._RESPONSE_BANK = None
        mod._load_response_bank(str(BANK_PATH))

        # Supported case
        ok, missing = mod.gpt2_indicators_supported_by_bank(
            ["EMO", "URG"], "fake", "novice")
        assert ok is True
        assert missing == []

        # Unsupported case (made-up indicator)
        ok2, missing2 = mod.gpt2_indicators_supported_by_bank(
            ["EMO", "FAKEIND"], "fake", "novice")
        assert ok2 is False
        assert "FAKEIND" in missing2

    def test_explanation_uses_bank_phrases(self):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "merge29_c", str(REPO_ROOT / "scripts" / "29_merge_gpt2_into_pool.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._RESPONSE_BANK = None
        mod._load_response_bank(str(BANK_PATH))

        exp = mod._build_explanation("fake", ["EMO"], "novice", card_idx=0)
        assert exp["bank_version"] != "stub"
        # Phrase should NOT be the v2.8.7 stub
        assert "GPT-2 generation flagged" not in exp["indicator_phrases"][0]["phrase"]
        # Phrase should be Tagalog-leading (contains common Tagalog words)
        phrase = exp["indicator_phrases"][0]["phrase"]
        assert any(w in phrase.lower() for w in ["ng", "ang", "sa", "na", "ay"])

    def test_explanation_diversity_across_cards(self):
        """Two cards with the same indicator should NOT get identical phrases
        if card_idx differs — that's how we push pool diversity above 30%."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "merge29_d", str(REPO_ROOT / "scripts" / "29_merge_gpt2_into_pool.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._RESPONSE_BANK = None
        mod._load_response_bank(str(BANK_PATH))

        phrases = set()
        for idx in range(6):
            exp = mod._build_explanation("fake", ["EMO"], "novice", card_idx=idx)
            phrases.add(exp["indicator_phrases"][0]["phrase"])
        # 3 variants in the bank → expect 3 distinct phrases across 6 cards
        assert len(phrases) >= 2, (
            f"Card-idx rotation should produce variant phrases, got {phrases}"
        )

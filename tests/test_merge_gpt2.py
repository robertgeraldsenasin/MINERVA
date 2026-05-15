"""Unit tests for v2.8.7 scripts/29_merge_gpt2_into_pool.py.

Verifies:
  - Sentence recovery from GPT-2 mid-word truncation
  - "Candidate <code>" remapping to A/B/C order-of-appearance
  - Drop cards with >3 distinct entities
  - Schema mapping produces template-shape cards
  - End-to-end merge against an empty-template baseline

Run:
    python -m pytest tests/test_merge_gpt2.py -v
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "29_merge_gpt2_into_pool.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("merge29", str(SCRIPT_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["merge29"] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load_module()


# recover_truncated_text

class TestRecoverTruncatedText:
    def test_already_clean_text_unchanged(self):
        t = "May halalan na sa Mayo. Marami ang naghahanda."
        out, status = m.recover_truncated_text(t)
        assert out == t
        assert status == "ok"

    def test_trims_to_last_period(self):
        t = ("Sinabi ng kalihim na ang halalan ay magaganap sa Mayo "
             "kasama ang mga LGU at COMELEC officials sa buong bansa. "
             "Pero ayon sa ulat, may mga kababayan natin ang nag-aalala "
             "tungkol sa")  # cut off mid-clause
        out, status = m.recover_truncated_text(t)
        assert out.endswith("bansa."), f"Expected to end with 'bansa.' but got: {out[-30:]!r}"
        assert status == "trimmed"

    def test_preserves_question_mark_endings(self):
        t = ("Saan tayo pupunta sa darating na bakasyon ng eskwela? "
             "May mga miyembro ng pamilya na may iba't ibang interes "
             "kaya magkakaroon ng diskusyon? Hindi ko alam talaga")
        out, status = m.recover_truncated_text(t)
        assert out.endswith("?"), f"Expected to end with '?' but got: {out[-30:]!r}"
        assert status == "trimmed"

    def test_unrecoverable_when_no_terminator(self):
        t = "Walang anumang puntuasyon na nasa loob ng tekstong ito at malayo pa"
        out, status = m.recover_truncated_text(t)
        assert status == "unrecoverable"

    def test_unrecoverable_when_terminator_too_early(self):
        t = "Hi. Then a long ramble follows without any further punctuation marker at all"
        out, status = m.recover_truncated_text(t, min_chars=80)
        # Period exists but only at position 2, way below 80 chars
        assert status == "unrecoverable"


# passes_quality_filter — handles dict-shaped truncation_flag

class TestQualityFilter:
    def test_accepts_recoverable_truncation(self):
        card = {
            "text": ("Sa pagbisita ng mga opisyal ng Candidate DKR sa "
                     "Candidate DKJ, muling pinaalalahanan ng Malacanang. "
                     "Ang mga residente ay nagulat sa balita."),
            "truncation_flag": {"is_truncated": True, "reason": "dangling_function_word"},
        }
        ok, reason, recovered = m.passes_quality_filter(card)
        assert ok is True
        # Should have trimmed back to the last "."
        assert recovered.endswith(".")

    def test_rejects_empty(self):
        ok, reason, recovered = m.passes_quality_filter({"text": ""})
        assert ok is False
        assert "empty" in reason

    def test_rejects_too_short_flag(self):
        card = {
            "text": "Maikli lang ito. Wala nang iba.",
            "truncation_flag": {"is_truncated": True, "reason": "too_short"},
        }
        ok, reason, _ = m.passes_quality_filter(card)
        assert ok is False
        assert "too_short" in reason


# remap_to_allowlist

class TestRemapToAllowlist:
    def test_remaps_two_distinct_codes(self):
        t = "Pagbisita ni Candidate DKR kay Candidate JZQ kahapon."
        out, n, mp = m.remap_to_allowlist(t)
        assert "Candidate A" in out
        assert "Candidate B" in out
        assert "Candidate DKR" not in out
        assert "Candidate JZQ" not in out
        assert n == 2
        assert mp == {"DKR": "A", "JZQ": "B"}

    def test_remaps_three_distinct_codes(self):
        t = "Sina Candidate XY, Candidate AB, at Candidate Z ay nagkita."
        out, n, mp = m.remap_to_allowlist(t)
        assert mp == {"XY": "A", "AB": "B", "Z": "C"}
        assert "Candidate A" in out and "Candidate B" in out and "Candidate C" in out
        assert n == 3

    def test_drops_when_more_than_three_distinct(self):
        t = ("Sina Candidate AA, Candidate BB, Candidate CC, "
             "at Candidate DD ay nagkita.")
        out, n, mp = m.remap_to_allowlist(t)
        assert n == 4
        assert mp == {}  # signals "drop this card"

    def test_preserves_existing_a_b_c(self):
        t = "Si Candidate A at Candidate B ay magkapatid."
        out, n, mp = m.remap_to_allowlist(t)
        assert mp == {"A": "A", "B": "B"}
        assert out == t  # text unchanged

    def test_collapses_repeated_mentions(self):
        t = ("Si Candidate XYZ ay sumagot. Si Candidate XYZ ay "
             "nagsalita rin. Si Candidate ABC ay tumango.")
        out, n, mp = m.remap_to_allowlist(t)
        assert n == 2  # 2 distinct codes
        assert out.count("Candidate A") == 2  # both XYZ mentions
        assert out.count("Candidate B") == 1  # the one ABC mention

    def test_no_candidate_returns_empty_mapping(self):
        out, n, mp = m.remap_to_allowlist("Walang anumang Candidate dito.")
        assert n == 0
        assert mp == {}


# passes_candidate_filter — runs after remap

class TestCandidateFilter:
    def test_passes_after_remap(self):
        ok, _ = m.passes_candidate_filter(
            "Si Candidate A at Candidate B ay nag-usap.")
        assert ok is True

    def test_rejects_no_reference(self):
        ok, reason = m.passes_candidate_filter("Walang Candidate dito.")
        assert ok is False
        assert "no_candidate_reference" in reason

    def test_rejects_foreign_code_post_remap(self):
        # Defensive — if remap was skipped somehow
        ok, reason = m.passes_candidate_filter("Si Candidate DKR ay sumagot.")
        assert ok is False
        assert "foreign_candidate_codes" in reason


# gpt2_card_to_template_shape

class TestSchemaMapping:
    def test_produces_all_template_keys(self):
        g = {
            "target": "fake",
            "control_tokens": {"label": "fake", "graph": "high",
                               "qlat": "high", "ensem": "high",
                               "tier": "novice"},
            "text": "Si Candidate A ay umano'y nagnakaw ng pera.",
            "named_features": {"ind_emo_fired": 1.0, "ind_miss_fired": 1.0,
                               "ind_emo_score": 0.7},
            "detectors": {"p_roberta_fake": 0.85, "p_distil_fake": 0.78,
                          "p_degnn_fake": 0.81, "p_ensemble_fake": 0.80},
            "p_fake": 0.81,
            "truncation_flag": False,
        }
        card = m.gpt2_card_to_template_shape(g, idx=0)
        # Required keys for downstream scripts (31, 23, 24, 26, 32, 33)
        for key in ["id", "text", "candidate", "target_label", "verdict",
                    "fake_likelihood_percent", "credibility_percent",
                    "difficulty_bin", "fired_indicators", "indicator_details",
                    "named_features", "qlattice", "detectors", "heuristics",
                    "theme_flags", "explanation", "provenance", "metadata"]:
            assert key in card, f"Missing required field: {key}"

    def test_verdict_from_pfake(self):
        for p, expected in [(0.81, "FAKE"), (0.50, "UNCERTAIN"),
                            (0.20, "REAL"), (0.65, "FAKE"), (0.34, "REAL")]:
            g = {"target": "fake", "text": "Si Candidate A ay sumagot.",
                 "p_fake": p, "named_features": {}, "detectors": {},
                 "control_tokens": {"tier": "novice"}, "truncation_flag": False}
            card = m.gpt2_card_to_template_shape(g, idx=0)
            assert card["verdict"] == expected, f"p_fake={p} expected {expected}"

    def test_provenance_marks_gpt2_source(self):
        g = {"target": "fake", "text": "Si Candidate A ay sumagot.",
             "p_fake": 0.5, "named_features": {}, "detectors": {},
             "control_tokens": {}, "truncation_flag": False}
        card = m.gpt2_card_to_template_shape(g, idx=0)
        assert card["provenance"]["source"] == "gpt2_neurosymbolic"
        assert card["provenance"]["generator"] == "gpt2_neurosymbolic"
        assert "29_merge_gpt2_into_pool" in card["provenance"]["script_chain"]

    def test_candidate_field_extracted_from_text(self):
        g = {"target": "real", "text": "Si Candidate B ay nagsalita.",
             "p_fake": 0.2, "named_features": {}, "detectors": {},
             "control_tokens": {}, "truncation_flag": False}
        card = m.gpt2_card_to_template_shape(g, idx=0)
        assert card["candidate"] == "C-B"


# End-to-end merge

class TestEndToEndMerge:
    def test_merges_templates_with_promoted_gpt2(self, tmp_path):
        # Templates
        templates = [
            {"id": "tpl_001", "text": "Si Candidate A ay totoo.",
             "candidate": "C-A", "verdict": "REAL", "target_label": "real",
             "fake_likelihood_percent": 12.0, "credibility_percent": 88.0,
             "difficulty_bin": "easy", "fired_indicators": ["MISS"],
             "indicator_details": {}, "named_features": {},
             "qlattice": {}, "detectors": {}, "heuristics": {},
             "theme_flags": {}, "explanation": {},
             "provenance": {"source": "template_v2.6"}, "metadata": {}},
        ]
        templates_path = tmp_path / "templates.json"
        templates_path.write_text(json.dumps(templates), encoding="utf-8")

        # GPT-2 fake (one good, one with too many entities, one truncated badly)
        fake_jsonl = tmp_path / "fake.jsonl"
        with open(fake_jsonl, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "target": "fake",
                "text": ("Sinasabi ng mga supporter ni Candidate XYZ na "
                         "ipinagkakanulo ng kabilang grupo ang halalan. "
                         "Walang ebidensya na ipinakita ang mga ito."),
                "named_features": {"ind_miss_fired": 1.0},
                "detectors": {"p_roberta_fake": 0.78},
                "p_fake": 0.78,
                "truncation_flag": {"is_truncated": False},
                "control_tokens": {"tier": "novice"},
            }) + "\n")
            # Too many entities
            f.write(json.dumps({
                "target": "fake",
                "text": ("Sina Candidate AA, Candidate BB, Candidate CC, "
                         "Candidate DD, at Candidate EE ay nag-usap. "
                         "Marami silang pinag-usapan."),
                "named_features": {}, "detectors": {}, "p_fake": 0.5,
                "control_tokens": {}, "truncation_flag": False,
            }) + "\n")

        # Empty real
        real_jsonl = tmp_path / "real.jsonl"
        real_jsonl.touch()

        out_path = tmp_path / "merged.json"
        report_path = tmp_path / "report.json"

        report = m.merge(
            templates_path=templates_path,
            gpt2_fake_path=fake_jsonl,
            gpt2_real_path=real_jsonl,
            out_path=out_path,
            report_path=report_path,
        )

        assert report["templates_in"] == 1
        assert report["gpt2_fake_attempted"] == 2
        assert report["gpt2_promoted_total"] == 1
        assert report["gpt2_rejected_total"] == 1
        assert report["merged_total"] == 2

        merged = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(merged) == 2
        # First card is the template, second is the GPT-2 promotion
        assert merged[0]["id"] == "tpl_001"
        assert merged[1]["provenance"]["source"] == "gpt2_neurosymbolic"
        # XYZ should have been remapped to A
        assert "Candidate A" in merged[1]["text"]
        assert "Candidate XYZ" not in merged[1]["text"]


# Static check: notebook references the merge script

class TestScriptExistence:
    def test_script_29_is_executable(self):
        src = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "def main()" in src
        assert "argparse" in src
        assert "remap_to_allowlist" in src
        assert "recover_truncated_text" in src

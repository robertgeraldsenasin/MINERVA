"""
Unit tests for minerva_filters.

Run:
    python -m pytest tests/test_filters.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from minerva_filters import (keyword_score, is_truncated,
                              has_legacy_pseudonyms,
                              mentions_one_of_three,
                              run_all_gates)


def test_keyword_score_high_for_electoral():
    s, _ = keyword_score("Sen. Marquez filed a campaign-finance bill before the election.")
    assert s >= 0.55


def test_keyword_score_low_for_grab_leak():
    s, _ = keyword_score("Walang Grab fare hike ngayong linggo.")
    assert s < 0.55


def test_keyword_score_low_for_meralco_leak():
    s, _ = keyword_score("Meralco electricity bill increase next month.")
    assert s < 0.55


def test_is_truncated_no_terminal():
    truncated, why = is_truncated(
        "Sen. Marquez vowed to push for new infrastructure spending next "
        "quarter, citing the need for")
    assert truncated
    assert why == "no_terminal"


def test_is_truncated_dangling_function_word():
    truncated, why = is_truncated("Sen. Marquez vowed to push for new infrastructure with.")
    assert truncated
    assert why == "dangling_function_word"


def test_is_truncated_complete_text():
    truncated, why = is_truncated("Sen. Marquez vowed to push for new infrastructure.")
    assert not truncated


def test_has_legacy_pseudonym_caught():
    found, items = has_legacy_pseudonyms("Candidate DTQ said something today.")
    assert found
    assert "Candidate DTQ" in items


def test_has_legacy_pseudonym_not_present():
    found, _ = has_legacy_pseudonyms(
        "Sen. Reynaldo \"Rey\" Marquez said something today.")
    assert not found


def test_mentions_one_of_three_marquez():
    ok, code = mentions_one_of_three(
        "Sen. Reynaldo \"Rey\" Marquez announced his platform.", [])
    assert ok
    assert code == "C-RM"


def test_mentions_one_of_three_none():
    ok, code = mentions_one_of_three("This post mentions no candidate.", [])
    assert not ok


def test_run_all_gates_accepts_clean_card():
    txt = ("Vice-Mayor Iris Bantayan filed a transparency bill at the Senate. "
           "Full text at https://www.senate.gov.ph/bill-1234.")
    r = run_all_gates(txt)
    assert r.accepted, f"Should accept, got reasons: {r.reasons}"


def test_run_all_gates_rejects_grab_leak():
    txt = "Hindi tutuloy ng Grab ang fare hike ngayong linggo."
    r = run_all_gates(txt)
    assert not r.accepted
    assert any("theme" in reason for reason in r.reasons)


def test_run_all_gates_rejects_legacy_pseudonym():
    txt = ("Sinabi ni Candidate DTQ na siya ang kakampi sa halalan. "
           "Sources say agad-agad.")
    r = run_all_gates(txt)
    assert not r.accepted
    assert any("legacy_pseudonyms" in reason for reason in r.reasons)


def test_run_all_gates_rejects_truncated():
    txt = ("Sen. Reynaldo \"Rey\" Marquez vowed to push for new infrastructure "
           "funding next quarter, citing the need for")
    r = run_all_gates(txt)
    assert not r.accepted
    assert any("truncation" in reason for reason in r.reasons)


def test_run_all_gates_diagnostics_present():
    r = run_all_gates("A short test post about Marquez and policy.")
    assert "theme_score" in r.diagnostics
    assert "is_electoral" in r.diagnostics

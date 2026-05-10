"""
Unit tests for scripts/33_strict_name_allowlist.py (v2.6.final).

Run:
    python -m pytest tests/test_strict_allowlist.py -v
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "33_strict_name_allowlist.py"


def _load_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "strict_allowlist", str(SCRIPT_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load_module()


@pytest.fixture
def candidate_profiles_path(tmp_path):
    """Generic Candidate A/B/C profiles."""
    profiles = {
        "A": {
            "candidate_id": "A",
            "display_name": "Candidate A",
            "public_name": "Candidate A",
            "short_name": "Candidate A",
            "aliases": ["Candidate A", "candidate A", "Candidate-A"],
        },
        "B": {
            "candidate_id": "B",
            "display_name": "Candidate B",
            "public_name": "Candidate B",
            "short_name": "Candidate B",
            "aliases": ["Candidate B", "candidate B", "Candidate-B"],
        },
        "C": {
            "candidate_id": "C",
            "display_name": "Candidate C",
            "public_name": "Candidate C",
            "short_name": "Candidate C",
            "aliases": ["Candidate C", "candidate C", "Candidate-C"],
        },
    }
    p = tmp_path / "candidate_profiles.json"
    p.write_text(json.dumps(profiles), encoding="utf-8")
    return p


@pytest.fixture
def blocklist_path(tmp_path):
    p = tmp_path / "blocklist.txt"
    p.write_text(
        "# test blocklist\n"
        "marcos\nduterte\nrobredo\npacquiao\naquino\n"
        "ferdinand marcos\nrodrigo duterte\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def allowed(candidate_profiles_path):
    return m.load_allowlist(candidate_profiles_path)


@pytest.fixture
def blocked(blocklist_path):
    return m.load_blocklist(blocklist_path)


# === detect_person_spans ===

def test_detect_title_plus_single_name():
    text = "Sen. Marcos commented yesterday."
    spans = m.detect_person_spans(text)
    names = [s[2] for s in spans]
    assert "Marcos" in names


def test_detect_according_to_attribution():
    text = "According to Robredo, the issue is unresolved."
    spans = m.detect_person_spans(text)
    names = [s[2] for s in spans]
    assert "Robredo" in names


def test_skip_definite_non_names():
    text = "Breaking news today: the council met."
    spans = m.detect_person_spans(text)
    names = [s[2] for s in spans]
    assert "Breaking" not in names


def test_skip_tagalog_function_words():
    text = "Pero ang report niya, walang pruweba."
    spans = m.detect_person_spans(text)
    names = [s[2] for s in spans]
    assert "Pero" not in names
    assert "Walang" not in names


def test_skip_atty_alone():
    """'Atty' standalone is a title, not a name."""
    text = "Atty. is short for attorney."
    spans = m.detect_person_spans(text)
    names = [s[2] for s in spans]
    assert "Atty" not in names


# === detect_blocklist_tokens ===

def test_blocklist_direct_naked_surname(blocked):
    text = "Pacquiao endorses the candidate."
    hits = m.detect_blocklist_tokens(text, blocked)
    names = [h[2] for h in hits]
    assert any(n.lower() == "pacquiao" for n in names)


def test_blocklist_case_insensitive(blocked):
    for variant in ["Marcos", "MARCOS", "marcos"]:
        text = f"{variant} statement."
        hits = m.detect_blocklist_tokens(text, blocked)
        assert len(hits) >= 1, f"failed for {variant!r}"


# === load_allowlist ===

def test_allowlist_includes_generic_codes(allowed):
    assert "candidate a" in allowed
    assert "candidate b" in allowed
    assert "candidate c" in allowed


def test_allowlist_does_not_include_blocked_names(allowed):
    assert "marcos" not in allowed
    assert "duterte" not in allowed


# === REJECT mode ===

def test_reject_mode_drops_leaky_card(allowed, blocked):
    card = {
        "id": "leak1",
        "text": "Sen. Marcos questioned Candidate A about the budget.",
        "verdict": "FAKE",
    }
    keep, _, foreign = m.process_card(
        card, allowed, blocked, "reject", "[X]", "text"
    )
    assert not keep
    assert any("marcos" in f[2].lower() for f in foreign)


def test_reject_mode_keeps_clean_card(allowed, blocked):
    card = {
        "id": "clean1",
        "text": "Candidate A addressed the city council yesterday.",
        "verdict": "REAL",
    }
    keep, _, foreign = m.process_card(
        card, allowed, blocked, "reject", "[X]", "text"
    )
    assert keep
    assert foreign == []


def test_reject_catches_naked_blocked_surname(allowed, blocked):
    """'Pacquiao' with no title must still be rejected via blocklist scan."""
    card = {
        "id": "leak2",
        "text": "Pacquiao endorses Candidate C in the upcoming poll.",
        "verdict": "FAKE",
    }
    keep, _, _ = m.process_card(
        card, allowed, blocked, "reject", "[X]", "text"
    )
    assert not keep


def test_reject_keeps_card_with_real_factchecker(allowed, blocked):
    """Vera Files is on _ALLOWED_ORGANIZATIONS."""
    card = {
        "id": "factcheck1",
        "text": "Vera Files reported that the claim about Candidate B is false.",
        "verdict": "REAL",
    }
    keep, _, foreign = m.process_card(
        card, allowed, blocked, "reject", "[X]", "text"
    )
    assert keep, f"unexpectedly rejected: {foreign}"


def test_reject_drops_invented_first_names(allowed, blocked):
    """'Aurelia Santos' is no longer allowed (generic-only policy)."""
    card = {
        "id": "leak3",
        "text": "Aurelia Santos held a rally yesterday.",
        "verdict": "REAL",
    }
    keep, _, foreign = m.process_card(
        card, allowed, blocked, "reject", "[X]", "text"
    )
    assert not keep, "generic-only policy means Aurelia Santos must be flagged"


# === REDACT mode ===

def test_redact_mode_replaces_foreign_names(allowed, blocked):
    card = {
        "id": "leak4",
        "text": "Sen. Marcos and Candidate A disagreed.",
        "verdict": "FAKE",
    }
    keep, modified, _ = m.process_card(
        card, allowed, blocked, "redact", "[Iba]", "text"
    )
    assert keep
    assert "Marcos" not in modified["text"]
    assert "Candidate A" in modified["text"]
    assert "[Iba]" in modified["text"]
    assert modified["strict_allowlist_redactions"] >= 1


def test_redact_handles_overlapping_spans(allowed, blocked):
    """'Mayor Duterte' must redact cleanly without nested artifacts."""
    card = {
        "id": "leak5",
        "text": "Mayor Duterte and Candidate A met yesterday.",
        "verdict": "FAKE",
    }
    keep, modified, _ = m.process_card(
        card, allowed, blocked, "redact", "[X]", "text"
    )
    assert keep
    assert "[X]te" not in modified["text"]
    assert "[X][X]" not in modified["text"]


# === Allowed names never flagged ===

def test_candidate_a_never_flagged(allowed, blocked):
    text = "Candidate A signed the bill."
    spans = m.detect_person_spans(text)
    _, foreign = m.classify_spans(spans, allowed, blocked)
    assert foreign == [], f"unexpectedly flagged: {foreign}"


def test_candidate_c_never_flagged(allowed, blocked):
    text = "Today Candidate C filed a statement."
    spans = m.detect_person_spans(text)
    _, foreign = m.classify_spans(spans, allowed, blocked)
    assert foreign == [], f"unexpectedly flagged: {foreign}"


def test_quezon_city_never_flagged(allowed, blocked):
    """Multi-word PH place names must not be flagged."""
    card = {
        "id": "loc1",
        "text": "The event was held in Quezon City yesterday.",
        "verdict": "REAL",
    }
    keep, _, foreign = m.process_card(
        card, allowed, blocked, "reject", "[X]", "text"
    )
    assert keep, f"unexpectedly rejected: {foreign}"


def test_pulse_asia_never_flagged(allowed, blocked):
    """Real polling firms must not be flagged."""
    card = {
        "id": "poll1",
        "text": "Pulse Asia released its latest survey.",
        "verdict": "REAL",
    }
    keep, _, foreign = m.process_card(
        card, allowed, blocked, "reject", "[X]", "text"
    )
    assert keep, f"unexpectedly rejected: {foreign}"

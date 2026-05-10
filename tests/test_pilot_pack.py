"""
Unit tests for scripts/40_export_pilot_pack.py.

Run:
    python -m pytest tests/test_pilot_pack.py -v
"""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "40_export_pilot_pack.py"


def _load_pilot_module():
    """Import a script whose filename starts with a digit via importlib."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "export_pilot_pack", str(SCRIPT_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pilot = _load_pilot_module()


def _make_card(idx: int, verdict: str, tier: str,
               candidate: str, tactic: str) -> dict:
    return {
        "id": f"tpl_{tactic}_{candidate}_{idx:05d}",
        "text": f"Sample post {idx} on {candidate} using {tactic}.",
        "verdict": verdict,
        "candidate": candidate,
        "fired_indicators": ["MISS"] if verdict == "FAKE" else [],
        "explanation": {
            "summary": f"Card {idx} summary.",
            "indicator_phrases": [{
                "indicator": "MISS" if verdict != "REAL" else "CREDIBLE",
                "phrase": f"Justifying phrase for card {idx}.",
            }],
        },
        "provenance": {"tactic": tactic, "tier": tier},
    }


def _mock_pool(n: int = 50) -> list[dict]:
    """50-card mock pool with pool-like ratios:
    37 FAKE / 8 REAL / 5 UNCERTAIN, 3 candidates, 3 tiers, 5 tactics.
    """
    verdicts = ["FAKE"] * 37 + ["REAL"] * 8 + ["UNCERTAIN"] * 5
    tiers = ["novice", "proficient", "advanced"]
    candidates = ["C-A", "C-B", "C-C"]
    tactics = ["urgency_sharing", "red_tagging", "conspiracy_theory",
               "credible_policy_announcement", "discrediting_personal_attack"]
    return [
        _make_card(
            i,
            verdict=verdicts[i],
            tier=tiers[i % 3],
            candidate=candidates[i % 3],
            tactic=tactics[i % 5],
        )
        for i in range(n)
    ]


def test_pilot_pack_sampling_and_export(tmp_path):
    """Single integration test for HANDOFF.md P1.2 sampler + writers.

    Covers (per task spec): correct sample size, reproducibility,
    proportional verdict ratios, >=3 tactics covered, >=3 candidates
    covered, and all three output files written with expected
    content markers.
    """
    pool = _mock_pool(50)

    sample = pilot.stratified_sample(pool, n=30, seed=1729)

    # Reproducibility: same seed -> same sample.
    sample_again = pilot.stratified_sample(pool, n=30, seed=1729)
    assert [c["id"] for c in sample] == [c["id"] for c in sample_again]

    # Sample size.
    assert len(sample) == 30

    # Verdict ratio (proportional to pool: 37/8/5 over 50 -> 22/5/3 over 30).
    verdicts = Counter(c["verdict"] for c in sample)
    assert 20 <= verdicts.get("FAKE", 0) <= 24, verdicts
    assert 4 <= verdicts.get("REAL", 0) <= 7, verdicts
    assert 2 <= verdicts.get("UNCERTAIN", 0) <= 4, verdicts

    # Tactic coverage: at least 3 of the 5 mock tactics.
    tactics = {pilot.get_tactic(c) for c in sample}
    assert len(tactics) >= 3, f"only {len(tactics)} tactics: {tactics}"

    # Candidate coverage: all 3 candidates represented.
    cands = {c["candidate"] for c in sample}
    assert len(cands) >= 3, f"only {len(cands)} candidates: {cands}"

    # Three output files exist and have non-trivial content.
    tactic_options = sorted({pilot.get_tactic(c) for c in pool})
    html_path = tmp_path / "printable_card_pack.html"
    quest_path = tmp_path / "questionnaire.md"
    key_path = tmp_path / "answer_key.md"

    pilot.write_html(sample, html_path)
    pilot.write_questionnaire(sample, tactic_options, quest_path)
    pilot.write_answer_key(sample, key_path)

    assert html_path.exists() and html_path.stat().st_size > 1000
    assert quest_path.exists() and quest_path.stat().st_size > 500
    assert key_path.exists() and key_path.stat().st_size > 500

    # Questionnaire shows all 5 question labels.
    qtext = quest_path.read_text(encoding="utf-8")
    for marker in ("**Q1.", "**Q2.", "**Q3.", "**Q4.", "**Q5."):
        assert marker in qtext, f"missing {marker} in questionnaire"

    # Answer key reveals the gold verdict and a justifying phrase.
    ktext = key_path.read_text(encoding="utf-8")
    assert "Gold verdict" in ktext
    assert "Justifying phrase" in ktext

    # HTML pack has page-break CSS and one card section per sampled card.
    htext = html_path.read_text(encoding="utf-8")
    assert "page-break-after" in htext
    assert htext.count("<section class='card'>") == len(sample)

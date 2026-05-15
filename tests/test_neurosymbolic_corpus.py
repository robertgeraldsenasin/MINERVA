"""
Unit tests for scripts/10b_prepare_gpt2_neurosymbolic.py (v2.6.final).

Tests the helper functions in isolation so we don't need transformers,
torch, datasets, or HF Hub access at test time.

Run:
    python -m pytest tests/test_neurosymbolic_corpus.py -v
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "10b_prepare_gpt2_neurosymbolic.py"


def _load_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "neurosymbolic_corpus", str(SCRIPT_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load_module()


# Bin function — bin3

def test_bin3_high_value_returns_high():
    out = m.bin3(0.9, 0.6, 0.8, "HIGH", "MID", "LOW", "UNK")
    assert out == "HIGH"


def test_bin3_mid_value_returns_mid():
    out = m.bin3(0.7, 0.6, 0.8, "HIGH", "MID", "LOW", "UNK")
    assert out == "MID"


def test_bin3_low_value_returns_low():
    out = m.bin3(0.3, 0.6, 0.8, "HIGH", "MID", "LOW", "UNK")
    assert out == "LOW"


def test_bin3_none_returns_unk():
    out = m.bin3(None, 0.6, 0.8, "HIGH", "MID", "LOW", "UNK")
    assert out == "UNK"


def test_bin3_nan_returns_unk():
    out = m.bin3(float("nan"), 0.6, 0.8, "HIGH", "MID", "LOW", "UNK")
    assert out == "UNK"


def test_bin3_at_high_threshold_inclusive():
    """0.8 == high threshold should bin as HIGH."""
    out = m.bin3(0.8, 0.6, 0.8, "HIGH", "MID", "LOW", "UNK")
    assert out == "HIGH"


def test_bin3_at_low_threshold_inclusive():
    """0.6 == low threshold should bin as MID, not LOW."""
    out = m.bin3(0.6, 0.6, 0.8, "HIGH", "MID", "LOW", "UNK")
    assert out == "MID"


# Tier mapping — tier_from_margin

def test_tier_clear_case_is_novice():
    """A very confident model output (p_qlattice ~ 0.95) → easy
    teaching case → novice tier."""
    out = m.tier_from_margin(0.95)
    assert out == m.TIER_NOVICE


def test_tier_uncertain_is_advanced():
    """A model output near the decision boundary → ambiguous → advanced."""
    out = m.tier_from_margin(0.51)
    assert out == m.TIER_ADVANCED


def test_tier_moderate_is_proficient():
    """Middle-margin → proficient (some practice but not extreme)."""
    out = m.tier_from_margin(0.75)
    assert out == m.TIER_PROFICIENT


def test_tier_none_is_unk():
    out = m.tier_from_margin(None)
    assert out == m.TIER_UNK


def test_tier_symmetric():
    """Margin works on both sides of 0.5 (REAL and FAKE)."""
    fake_clear = m.tier_from_margin(0.95)  # very fake → novice
    real_clear = m.tier_from_margin(0.05)  # very real → novice
    assert fake_clear == real_clear == m.TIER_NOVICE


# QLattice equation evaluator

def test_evaluate_qlattice_simple_equation():
    """Evaluate a simple equation matching the existing pipeline shape."""
    df = pd.DataFrame({
        "p_roberta_fake": [0.9, 0.1, 0.5],
        "p_distil_fake":  [0.85, 0.15, 0.5],
    })
    eq = "logreg(p_roberta_fake + p_distil_fake - 1.0)"
    out = m.evaluate_qlattice(eq, df)
    assert len(out) == 3
    assert 0 <= out[0] <= 1
    assert out[0] > out[2] > out[1]  # 0.9+0.85 > 0.5+0.5 > 0.1+0.15


def test_evaluate_qlattice_handles_sanitized_names():
    """Equation often uses 'rpca0' instead of 'r_pca_0'. Both should work."""
    df = pd.DataFrame({
        "r_pca_0": [1.0, 2.0],
        "d_pca_1": [0.5, 1.5],
    })
    # Sanitized form (matches feyn output)
    eq = "logreg(rpca0 + dpca1)"
    out = m.evaluate_qlattice(eq, df)
    assert len(out) == 2
    assert 0 <= out[0] <= 1


def test_evaluate_qlattice_missing_variable_returns_nan():
    """Equation references a column that doesn't exist."""
    df = pd.DataFrame({"a": [1.0, 2.0]})
    eq = "missing_variable + a"
    out = m.evaluate_qlattice(eq, df)
    assert len(out) == 2
    assert np.all(np.isnan(out))


def test_evaluate_qlattice_empty_equation_returns_nan():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out = m.evaluate_qlattice("", df)
    assert np.all(np.isnan(out))


# row_to_line — line format

def test_row_to_line_label_fake():
    line = m.row_to_line(
        "test text", 1,
        graph_tok=m.GRAPH_HIGH, qlat_tok=m.QLAT_HIGH,
        ensem_tok=m.ENSEM_HIGH, tier_tok=m.TIER_NOVICE,
    )
    assert m.LABEL_FAKE in line
    assert m.GRAPH_HIGH in line
    assert m.QLAT_HIGH in line
    assert m.ENSEM_HIGH in line
    assert m.TIER_NOVICE in line
    assert "test text" in line


def test_row_to_line_label_real():
    line = m.row_to_line(
        "real text", 0,
        graph_tok=m.GRAPH_LOW, qlat_tok=m.QLAT_MID,
        ensem_tok=m.ENSEM_LOW, tier_tok=m.TIER_ADVANCED,
    )
    assert m.LABEL_REAL in line
    assert m.LABEL_FAKE not in line


def test_row_to_line_strips_newlines():
    line = m.row_to_line(
        "text with\nnewline", 1,
        graph_tok=m.GRAPH_UNK, qlat_tok=m.QLAT_UNK,
        ensem_tok=m.ENSEM_UNK, tier_tok=m.TIER_UNK,
    )
    assert "\n" not in line.split(m.TIER_UNK)[1]


# Special token registry

def test_all_special_tokens_count():
    """v2.6.final should declare exactly 18 special tokens:
    2 label + 4 graph + 4 qlat + 4 ensem + 4 tier."""
    assert len(m.ALL_SPECIAL_TOKENS) == 18


def test_special_tokens_unique():
    assert len(m.ALL_SPECIAL_TOKENS) == len(set(m.ALL_SPECIAL_TOKENS))


def test_special_tokens_format():
    """Every special token follows <|key=value|> format."""
    import re
    pat = re.compile(r"^<\|[a-z]+=[a-z]+\|>$")
    for tok in m.ALL_SPECIAL_TOKENS:
        assert pat.match(tok), f"bad format: {tok!r}"


# build_corpus — small end-to-end with synthetic inputs

class _ArgsStub:
    """Minimal args stand-in for build_corpus."""
    graph_bins = (0.6, 0.8)
    qlat_bins = (0.6, 0.8)
    ensem_bins = (0.6, 0.8)


def test_build_corpus_smoke():
    """Build a corpus from synthetic inputs and verify the output shape."""
    raw = pd.DataFrame({
        "id": ["a", "b", "c"],
        "text": ["fake text 1", "real text 2", "fake text 3"],
        "label": [1, 0, 1],
    })
    feat = pd.DataFrame({
        "id": ["a", "b", "c"],
        "p_roberta_fake": [0.9, 0.1, 0.7],
        "p_distil_fake":  [0.85, 0.15, 0.65],
    })
    degnn = pd.DataFrame({
        "id": ["a", "b", "c"],
        "p_degnn_fake": [0.92, 0.08, 0.6],
    })
    eq = "logreg(p_roberta_fake + p_distil_fake - 1.0)"

    lines, bins, _thresholds = m.build_corpus(raw, feat, degnn, eq, _ArgsStub())

    # 3 lines emitted
    assert len(lines) == 3

    # Each line has all 5 control tokens
    for line in lines:
        assert any(t in line for t in (m.LABEL_FAKE, m.LABEL_REAL))
        assert any(t in line for t in (m.GRAPH_HIGH, m.GRAPH_MID,
                                       m.GRAPH_LOW, m.GRAPH_UNK))
        assert any(t in line for t in (m.QLAT_HIGH, m.QLAT_MID,
                                       m.QLAT_LOW, m.QLAT_UNK))
        assert any(t in line for t in (m.ENSEM_HIGH, m.ENSEM_MID,
                                       m.ENSEM_LOW, m.ENSEM_UNK))
        assert any(t in line for t in (m.TIER_NOVICE, m.TIER_PROFICIENT,
                                       m.TIER_ADVANCED, m.TIER_UNK))

    # Bin counts should sum to 3 for each signal
    assert sum(bins["graph"].values()) == 3
    assert sum(bins["qlat"].values()) == 3
    assert sum(bins["ensem"].values()) == 3
    assert sum(bins["tier"].values()) == 3
    assert bins["label"]["fake"] == 2
    assert bins["label"]["real"] == 1


def test_build_corpus_handles_missing_degnn():
    """Without DE-GNN preds, graph token should default to UNK."""
    raw = pd.DataFrame({
        "id": ["a"], "text": ["t"], "label": [1],
    })
    feat = pd.DataFrame({
        "id": ["a"],
        "p_roberta_fake": [0.9],
        "p_distil_fake": [0.85],
    })
    eq = "logreg(p_roberta_fake)"

    lines, bins, _thresholds = m.build_corpus(raw, feat, None, eq, _ArgsStub())

    assert len(lines) == 1
    assert m.GRAPH_UNK in lines[0]
    assert bins["graph"]["unk"] == 1


def test_build_corpus_handles_missing_qlattice():
    """Without QLattice equation, qlat token should default to UNK."""
    raw = pd.DataFrame({
        "id": ["a"], "text": ["t"], "label": [1],
    })
    feat = pd.DataFrame({
        "id": ["a"],
        "p_roberta_fake": [0.9],
        "p_distil_fake": [0.85],
    })
    degnn = pd.DataFrame({
        "id": ["a"],
        "p_degnn_fake": [0.92],
    })

    lines, bins, _thresholds = m.build_corpus(raw, feat, degnn, "", _ArgsStub())

    assert len(lines) == 1
    assert m.QLAT_UNK in lines[0]
    assert bins["qlat"]["unk"] == 1

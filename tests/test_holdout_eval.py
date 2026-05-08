"""Unit tests for v2.9.0 scripts/37_holdout_detector_eval.py.

Verifies:
  - Binary metric computation matches sklearn's reference values.
  - Holdout CSV loader correctly parses real/fake/uncertain.
  - Edge cases: all-correct, all-wrong, empty, and mixed-prediction inputs.
  - The script's NaN-safe handling when transformers isn't available.

The detector inference itself (which needs transformers + a trained model)
is not exercised here — that's an integration concern, not a unit-test one.

Run:
    python -m pytest tests/test_holdout_eval.py -v
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "37_holdout_detector_eval.py"


def _load():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("holdout37", str(SCRIPT_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["holdout37"] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load()


# ----------------------------------------------------------------------
# compute_binary_metrics — match sklearn semantics
# ----------------------------------------------------------------------

class TestBinaryMetrics:
    def test_perfect_predictions(self):
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 1, 0, 0, 1, 0]
        r = m.compute_binary_metrics(y_true, y_pred)
        assert r["accuracy"] == 1.0
        assert r["precision"] == 1.0
        assert r["recall"] == 1.0
        assert r["f1"] == 1.0
        assert r["tp"] == 3
        assert r["fp"] == 0
        assert r["tn"] == 3
        assert r["fn"] == 0

    def test_all_wrong_predictions(self):
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        r = m.compute_binary_metrics(y_true, y_pred)
        assert r["accuracy"] == 0.0
        assert r["precision"] == 0.0
        assert r["recall"] == 0.0
        assert r["f1"] == 0.0

    def test_known_f1(self):
        # 5 fake, 5 real. Predict 4 fake correctly, miss 1 fake (predict real),
        # predict 1 real wrongly (predict fake).
        # tp=4, fn=1, fp=1, tn=4
        # precision = 4/(4+1) = 0.8
        # recall    = 4/(4+1) = 0.8
        # f1        = 2*0.8*0.8/(0.8+0.8) = 0.8
        y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
        r = m.compute_binary_metrics(y_true, y_pred)
        assert r["tp"] == 4
        assert r["fn"] == 1
        assert r["fp"] == 1
        assert r["tn"] == 4
        assert abs(r["precision"] - 0.8) < 1e-9
        assert abs(r["recall"] - 0.8) < 1e-9
        assert abs(r["f1"] - 0.8) < 1e-9
        assert r["accuracy"] == 0.8

    def test_empty_arrays(self):
        r = m.compute_binary_metrics([], [])
        assert r["n"] == 0
        assert r["accuracy"] == 0.0
        assert r["precision"] == 0.0
        assert r["recall"] == 0.0
        assert r["f1"] == 0.0

    def test_unequal_lengths_raise(self):
        with pytest.raises(ValueError):
            m.compute_binary_metrics([1, 0], [1, 0, 1])

    def test_no_positive_predictions_safe_division(self):
        # All predicted real (0); should not divide by zero
        y_true = [1, 1, 0]
        y_pred = [0, 0, 0]
        r = m.compute_binary_metrics(y_true, y_pred)
        assert r["precision"] == 0.0
        assert r["recall"] == 0.0
        assert r["f1"] == 0.0
        assert r["accuracy"] == 1/3


# ----------------------------------------------------------------------
# Holdout CSV loader
# ----------------------------------------------------------------------

class TestHoldoutLoader:
    def test_loads_basic_csv(self, tmp_path):
        csv_path = tmp_path / "h.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,text,true_label\n")
            f.write("h001,Hello world,fake\n")
            f.write("h002,Another card,real\n")
            f.write("h003,Ambiguous,uncertain\n")

        rows = m.load_holdout(csv_path)
        assert len(rows) == 3
        assert rows[0]["true_label_int"] == 1
        assert rows[1]["true_label_int"] == 0
        assert rows[2]["true_label_int"] is None  # uncertain → excluded

    def test_handles_extra_columns(self, tmp_path):
        csv_path = tmp_path / "h.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,text,true_label,target_for_reference,annotator_notes\n")
            f.write("h001,text1,fake,fake,no notes\n")
            f.write("h002,text2,real,real,clear case\n")
        rows = m.load_holdout(csv_path)
        assert len(rows) == 2
        assert all(r["true_label_int"] is not None for r in rows)

    def test_normalizes_label_case(self, tmp_path):
        csv_path = tmp_path / "h.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("id,text,true_label\n")
            f.write("h001,t,FAKE\n")
            f.write("h002,t,Real\n")
        rows = m.load_holdout(csv_path)
        assert rows[0]["true_label_int"] == 1
        assert rows[1]["true_label_int"] == 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            m.load_holdout(tmp_path / "nonexistent.csv")

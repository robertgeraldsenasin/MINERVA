"""Unit tests for scripts/08_train_qlattice.py LogisticRegression fallback.

Verifies that when feyn is unavailable, the fallback emits a
`logreg(...)` equation that:
  - is syntactically parseable by minerva_qlattice.compile_equation
  - evaluates to (0,1) probabilities on training data
  - produces reasonable predictions (>50% accuracy on a separable problem)
  - works in the original (un-standardized) feature space

These tests run hermetically — no feyn import required.

Run:
    python -m pytest tests/test_qlattice_fallback.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_script_08():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "qlattice_train", str(SCRIPTS_DIR / "08_train_qlattice.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_qlattice_evaluator():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "minerva_qlattice", str(SCRIPTS_DIR / "minerva_qlattice.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["minerva_qlattice"] = mod
    spec.loader.exec_module(mod)
    return mod


s08 = _load_script_08()
qlmod = _load_qlattice_evaluator()


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def linearly_separable_df():
    """Build a small classification dataset that's almost-linearly separable."""
    rng = np.random.default_rng(42)
    n = 200
    # 4 PCA-like features + 2 detector probabilities
    feats = {
        "r_pca_0": rng.standard_normal(n),
        "r_pca_1": rng.standard_normal(n),
        "d_pca_0": rng.standard_normal(n),
        "d_pca_1": rng.standard_normal(n),
        "p_roberta_fake": rng.uniform(0, 1, n),
        "p_distil_fake": rng.uniform(0, 1, n),
    }
    df = pd.DataFrame(feats)
    # Label is a clear linear function of r_pca_0 and p_roberta_fake
    score = df["r_pca_0"] + 2.0 * df["p_roberta_fake"] - 1.0
    df["label"] = (score > 0).astype(int)
    return df


# ----------------------------------------------------------------------
# build_logreg_equation
# ----------------------------------------------------------------------

class TestBuildLogregEquation:
    def test_returns_string_with_logreg_wrapper(self, linearly_separable_df):
        feat_cols = [c for c in linearly_separable_df.columns if c != "label"]
        eq = s08.build_logreg_equation(linearly_separable_df, feat_cols)
        assert isinstance(eq, str)
        assert eq.startswith("logreg(")
        assert eq.endswith(")")

    def test_uses_sanitized_feature_names(self, linearly_separable_df):
        feat_cols = ["r_pca_0", "r_pca_1", "p_roberta_fake"]
        eq = s08.build_logreg_equation(linearly_separable_df, feat_cols)
        # Sanitized: r_pca_0 -> rpca0, p_roberta_fake -> probertafake
        assert "rpca0" in eq
        assert "rpca1" in eq
        assert "probertafake" in eq
        # The original underscore-form should NOT be in the equation
        # (the sanitizer strips them)
        assert "r_pca_0" not in eq

    def test_includes_all_features(self, linearly_separable_df):
        feat_cols = [c for c in linearly_separable_df.columns if c != "label"]
        eq = s08.build_logreg_equation(linearly_separable_df, feat_cols)
        for col in feat_cols:
            san = s08._sanitize_name(col)
            assert san in eq, f"Sanitized name {san!r} for {col!r} missing from equation"


# ----------------------------------------------------------------------
# Equation parses + evaluates via minerva_qlattice
# ----------------------------------------------------------------------

class TestEquationCompilesAndEvaluates:
    def test_equation_compiles(self, linearly_separable_df):
        feat_cols = [c for c in linearly_separable_df.columns if c != "label"]
        eq = s08.build_logreg_equation(linearly_separable_df, feat_cols)
        # Must compile without error
        code = qlmod.compile_equation(eq)
        assert code is not None

    def test_equation_evaluates_to_probability_range(self, linearly_separable_df):
        feat_cols = [c for c in linearly_separable_df.columns if c != "label"]
        eq = s08.build_logreg_equation(linearly_separable_df, feat_cols)

        code = qlmod.compile_equation(eq)
        bundle = qlmod.build_feature_locals(linearly_separable_df[feat_cols])
        out = qlmod.eval_compiled(code, bundle.locals, n_rows=len(linearly_separable_df))

        # Outputs must be valid probabilities
        assert out.shape == (len(linearly_separable_df),)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_predictions_correlate_with_labels(self, linearly_separable_df):
        """The fallback should produce predictions that beat random on a separable problem."""
        feat_cols = [c for c in linearly_separable_df.columns if c != "label"]
        eq = s08.build_logreg_equation(linearly_separable_df, feat_cols)

        code = qlmod.compile_equation(eq)
        bundle = qlmod.build_feature_locals(linearly_separable_df[feat_cols])
        probs = qlmod.eval_compiled(code, bundle.locals, n_rows=len(linearly_separable_df))

        preds = (probs > 0.5).astype(int)
        labels = linearly_separable_df["label"].astype(int).to_numpy()
        accuracy = (preds == labels).mean()

        # On a near-linearly-separable problem, accuracy should be >>50%
        assert accuracy > 0.85, (
            f"Fallback equation accuracy {accuracy:.2%} too low — "
            f"coefficient transform may be broken"
        )


# ----------------------------------------------------------------------
# Coefficient correctness — equation works in ORIGINAL feature space
# ----------------------------------------------------------------------

class TestCoefficientTransform:
    """The fallback fits LR on standardized features but emits an equation
    that operates on the original (un-standardized) features. Verify the
    coefficient transform is correct by comparing against sklearn's own
    predict_proba on the original space."""

    def test_equation_matches_sklearn_predict_proba(self, linearly_separable_df):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        feat_cols = [c for c in linearly_separable_df.columns if c != "label"]

        # Fit reference model the same way the fallback does
        X = linearly_separable_df[feat_cols].astype(float).to_numpy()
        y = linearly_separable_df["label"].astype(int).to_numpy()
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(Xs, y)
        sklearn_probs = clf.predict_proba(Xs)[:, 1]

        # Get probs via the equation path (which operates on un-scaled X)
        eq = s08.build_logreg_equation(linearly_separable_df, feat_cols)
        code = qlmod.compile_equation(eq)
        bundle = qlmod.build_feature_locals(linearly_separable_df[feat_cols])
        eq_probs = qlmod.eval_compiled(code, bundle.locals, n_rows=len(linearly_separable_df))

        # Should match very closely (modulo float precision on the
        # 6-significant-digit equation formatting)
        np.testing.assert_allclose(eq_probs, sklearn_probs, atol=1e-3)


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------

class TestEdgeCases:
    def test_handles_negative_coefficients(self):
        """Equation formatting must handle negative coeffs with proper - sign."""
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
        })
        # Label NEGATIVELY correlated with feat_a so coef is negative
        df["label"] = (df["feat_a"] < 0).astype(int)

        eq = s08.build_logreg_equation(df, ["feat_a", "feat_b"])
        # Equation must still be syntactically valid
        code = qlmod.compile_equation(eq)
        bundle = qlmod.build_feature_locals(df[["feat_a", "feat_b"]])
        out = qlmod.eval_compiled(code, bundle.locals, n_rows=len(df))
        assert (out >= 0).all() and (out <= 1).all()

    def test_handles_single_feature(self):
        """One-feature equation should still be valid."""
        rng = np.random.default_rng(0)
        n = 80
        df = pd.DataFrame({"only_feat": rng.standard_normal(n)})
        df["label"] = (df["only_feat"] > 0).astype(int)

        eq = s08.build_logreg_equation(df, ["only_feat"])
        assert "onlyfeat" in eq
        code = qlmod.compile_equation(eq)
        bundle = qlmod.build_feature_locals(df[["only_feat"]])
        out = qlmod.eval_compiled(code, bundle.locals, n_rows=len(df))
        assert out.shape == (len(df),)


# ----------------------------------------------------------------------
# Sanity: the script file imports the right names + has the loud-fail
# ----------------------------------------------------------------------

class TestScriptStructure:
    def test_script_does_not_silently_return_on_feyn_failure(self):
        """The v2.8 silent-return bug must NOT be back."""
        src = (SCRIPTS_DIR / "08_train_qlattice.py").read_text()
        # The pre-v2.8.4 version had: `OUT_NOTE.write_text(msg, ...); print(msg); return`
        # If that pattern returns to the script, downstream will silently fail again.
        assert "return\n" not in src.split("def main")[1].split("# Tier 1")[0], (
            "Found bare 'return' in main() before Tier 1 — silent-failure bug regressed"
        )

    def test_script_exposes_build_logreg_equation(self):
        """The fallback must be testable in isolation (helper exposed at module level)."""
        assert hasattr(s08, "build_logreg_equation")
        assert callable(s08.build_logreg_equation)

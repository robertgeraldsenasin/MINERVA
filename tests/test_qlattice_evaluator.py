"""Unit tests for scripts/minerva_qlattice.py — the QLattice safe evaluator.

Covers:
- _sigmoid numerical stability at extremes
- SAFE_FUNCS whitelist completeness
- sanitize_column_name behavior
- extract_variable_names parsing
- build_feature_locals aliasing
- compile_equation + eval_compiled determinism
- security: sandbox prevents dangerous calls

These tests are critical because the QLattice evaluator is the
deterministic explainability engine per BATB §1.4 SO 2.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
QL_PATH = REPO_ROOT / "scripts" / "minerva_qlattice.py"


@pytest.fixture(scope="module")
def qlmod():
    """Load minerva_qlattice as a module for testing."""
    # The qlattice module imports `pandas` and `numpy` at top level —
    # these are installed in test env, so direct import works.
    spec = importlib.util.spec_from_file_location("minerva_qlattice", QL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["minerva_qlattice"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# _sigmoid stability
# --------------------------------------------------------------------------

class TestSigmoid:
    def test_sigmoid_zero(self, qlmod):
        assert qlmod._sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_sigmoid_positive_extreme(self, qlmod):
        # Without clipping, exp(-1000) underflow → divide by zero possible.
        # With clipping at -60, sigmoid(60) ≈ 1.0 cleanly.
        result = qlmod._sigmoid(np.array([1000.0, 100.0, 60.0]))
        assert np.all(np.isfinite(result))
        assert np.all(result <= 1.0)
        assert np.all(result > 0.99)

    def test_sigmoid_negative_extreme(self, qlmod):
        result = qlmod._sigmoid(np.array([-1000.0, -100.0, -60.0]))
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert np.all(result < 0.01)

    def test_sigmoid_array_shape_preserved(self, qlmod):
        x = np.linspace(-10, 10, 50)
        out = qlmod._sigmoid(x)
        assert out.shape == x.shape


# --------------------------------------------------------------------------
# SAFE_FUNCS whitelist
# --------------------------------------------------------------------------

class TestSafeFuncs:
    def test_safe_funcs_includes_logreg_alias(self, qlmod):
        # 'logreg' is QLattice's name for sigmoid — must be aliased
        assert qlmod.SAFE_FUNCS["logreg"] is qlmod._sigmoid
        assert qlmod.SAFE_FUNCS["sigmoid"] is qlmod._sigmoid

    def test_safe_funcs_has_basic_math(self, qlmod):
        # Sanity check the whitelist
        for name in ("exp", "log", "log1p", "sqrt", "tanh", "sin", "cos", "abs"):
            assert name in qlmod.SAFE_FUNCS, f"missing {name}"

    def test_safe_funcs_has_constants(self, qlmod):
        assert qlmod.SAFE_FUNCS["pi"] == pytest.approx(np.pi)
        assert qlmod.SAFE_FUNCS["e"] == pytest.approx(np.e)

    def test_safe_funcs_min_max_aliases(self, qlmod):
        # Both 'min'/'max' and 'minimum'/'maximum' should resolve
        assert qlmod.SAFE_FUNCS["min"] is np.minimum
        assert qlmod.SAFE_FUNCS["max"] is np.maximum
        assert qlmod.SAFE_FUNCS["minimum"] is np.minimum
        assert qlmod.SAFE_FUNCS["maximum"] is np.maximum


# --------------------------------------------------------------------------
# Name handling
# --------------------------------------------------------------------------

class TestSanitizeColumnName:
    def test_sanitize_basic(self, qlmod):
        # Underscores stripped, alnum preserved
        assert qlmod.sanitize_column_name("r_pca_0") == "rpca0"
        assert qlmod.sanitize_column_name("d_pca_15") == "dpca15"

    def test_sanitize_leading_digit_prefixed(self, qlmod):
        # Names starting with digits get 'f' prefix to remain valid identifier
        result = qlmod.sanitize_column_name("0col")
        assert result.startswith("f")
        assert result == "f0col"

    def test_sanitize_special_chars(self, qlmod):
        assert qlmod.sanitize_column_name("foo-bar.baz") == "foobarbaz"
        assert qlmod.sanitize_column_name("a b c") == "abc"


class TestExtractVariableNames:
    def test_extract_simple(self, qlmod):
        names = qlmod.extract_variable_names("alpha + beta")
        assert names == {"alpha", "beta"}

    def test_extract_skips_safe_funcs(self, qlmod):
        # Known function names are filtered out
        names = qlmod.extract_variable_names("logreg(0.5 * rpca0 + dpca1)")
        assert "logreg" not in names
        assert names == {"rpca0", "dpca1"}

    def test_extract_handles_complex(self, qlmod):
        names = qlmod.extract_variable_names(
            "sigmoid(0.371 * dpca1 + 0.245 * rpca3 - 0.118)"
        )
        assert "sigmoid" not in names
        assert names == {"dpca1", "rpca3"}


# --------------------------------------------------------------------------
# build_feature_locals
# --------------------------------------------------------------------------

class TestBuildFeatureLocals:
    def test_build_basic(self, qlmod):
        df = pd.DataFrame({
            "r_pca_0": [0.1, 0.2, 0.3],
            "r_pca_1": [0.4, 0.5, 0.6],
            "label": [0, 1, 0],
        })
        bundle = qlmod.build_feature_locals(df)
        # Both original AND sanitized aliases registered
        assert "r_pca_0" in bundle.locals
        assert "rpca0" in bundle.locals
        # Aliases point back to original column name
        assert bundle.alias_to_source["rpca0"] == "r_pca_0"

    def test_build_skips_non_numeric(self, qlmod):
        df = pd.DataFrame({
            "x": [1.0, 2.0],
            "category": ["a", "b"],  # non-numeric
        })
        bundle = qlmod.build_feature_locals(df)
        assert "x" in bundle.locals
        assert "category" not in bundle.locals

    def test_build_includes_safe_funcs(self, qlmod):
        df = pd.DataFrame({"x": [1.0]})
        bundle = qlmod.build_feature_locals(df)
        assert "logreg" in bundle.locals
        assert bundle.locals["logreg"] is qlmod._sigmoid


# --------------------------------------------------------------------------
# Evaluation end-to-end
# --------------------------------------------------------------------------

class TestEvaluation:
    def test_compile_simple(self, qlmod):
        code = qlmod.compile_equation("0.5 * x + 1.0")
        out = qlmod.eval_compiled(code, {"x": np.array([0.0, 1.0, 2.0])}, 3)
        np.testing.assert_array_almost_equal(out, [1.0, 1.5, 2.0])

    def test_compile_caret_to_double_star(self, qlmod):
        # QLattice sometimes emits ^ for exponent — must accept
        code = qlmod.compile_equation("x^2")
        out = qlmod.eval_compiled(code, {"x": np.array([2.0, 3.0])}, 2)
        np.testing.assert_array_almost_equal(out, [4.0, 9.0])

    def test_compile_empty_raises(self, qlmod):
        with pytest.raises(ValueError):
            qlmod.compile_equation("")
        with pytest.raises(ValueError):
            qlmod.compile_equation("   ")

    def test_eval_qlattice_realistic_equation(self, qlmod):
        # Realistic logreg-style equation as it would be emitted by Feyn
        eq = "logreg(0.371093 * dpca1 + 0.245 * rpca0 - 0.118)"
        df = pd.DataFrame({
            "d_pca_1": [0.5, 1.0, 0.0],
            "r_pca_0": [0.2, 0.3, 0.1],
        })
        bundle = qlmod.build_feature_locals(df)
        code = qlmod.compile_equation(eq)
        out = qlmod.eval_compiled(code, bundle.locals, len(df))
        # Output must be in [0, 1] range (sigmoid output)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)
        assert out.shape == (3,)

    def test_eval_returns_array_for_constant(self, qlmod):
        # When the equation is just a constant, eval returns scalar; it
        # should be broadcast to shape (n_rows,)
        code = qlmod.compile_equation("0.5")
        out = qlmod.eval_compiled(code, {}, 5)
        assert out.shape == (5,)
        assert np.all(out == 0.5)


# --------------------------------------------------------------------------
# Sandbox security — the most important set
# --------------------------------------------------------------------------

class TestSandbox:
    def test_sandbox_blocks_builtins(self, qlmod):
        # Even though we use eval(), __builtins__ is suppressed
        code = qlmod.compile_equation("len([1,2,3])")
        with pytest.raises((NameError, TypeError)):
            qlmod.eval_compiled(code, {}, 1)

    def test_sandbox_blocks_import(self, qlmod):
        # __import__ is in builtins, not whitelisted
        code = qlmod.compile_equation("__import__('os')")
        with pytest.raises((NameError, TypeError)):
            qlmod.eval_compiled(code, {}, 1)

    def test_sandbox_blocks_open(self, qlmod):
        code = qlmod.compile_equation("open('/etc/passwd')")
        with pytest.raises((NameError, TypeError)):
            qlmod.eval_compiled(code, {}, 1)

    def test_sandbox_blocks_eval_within_eval(self, qlmod):
        code = qlmod.compile_equation("eval('1+1')")
        with pytest.raises((NameError, TypeError)):
            qlmod.eval_compiled(code, {}, 1)

    def test_sandbox_undefined_var_clean_error(self, qlmod):
        # An equation referencing an unknown variable should fail cleanly
        code = qlmod.compile_equation("nonexistent_feature * 2")
        with pytest.raises(NameError):
            qlmod.eval_compiled(code, {"x": np.array([1.0])}, 1)

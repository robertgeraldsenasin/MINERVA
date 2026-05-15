"""Unit tests for percentile binning in scripts/10b_prepare_gpt2_neurosymbolic.py.

The original v2.6.final fixed thresholds (0.6/0.8) produced ~96% of training
rows in the "high" bin on JCBlaise, because the detectors are well-separated
and most predictions land near 0 or 1. With ~96% imbalance, the GPT-2
control tokens become functionally identical — model can't learn what
<|qlat=low|> means if it sees <|qlat=high|> 97% of the time.

These tests prove that:
  - compute_percentile_thresholds(values) produces ~33/33/33 splits
  - On a JCBlaise-like distribution (96% above 0.8), percentile binning
    creates real contrast across low/mid/high
  - The fallback path (small samples, degenerate distributions) still works
  - tier-margin percentile thresholds work the same way

Run:
    python -m pytest tests/test_percentile_binning.py -v
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "10b_prepare_gpt2_neurosymbolic.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "neuro_corpus", str(SCRIPT_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["neuro_corpus"] = mod
    spec.loader.exec_module(mod)
    return mod


m = _load_module()


# compute_percentile_thresholds

class TestComputePercentileThresholds:
    def test_returns_low_below_high(self):
        values = list(np.linspace(0, 1, 100))
        t_low, t_high = m.compute_percentile_thresholds(values)
        assert t_low < t_high

    def test_uniform_distribution_gives_third_splits(self):
        """Uniform [0,1] should yield ~0.33, ~0.67 thresholds."""
        rng = np.random.default_rng(0)
        values = rng.uniform(0, 1, 10_000)
        t_low, t_high = m.compute_percentile_thresholds(values)
        assert 0.30 < t_low < 0.36, f"33rd percentile of uniform should be ~0.33, got {t_low}"
        assert 0.64 < t_high < 0.70, f"67th percentile of uniform should be ~0.67, got {t_high}"

    def test_jcblaise_like_distribution_creates_real_contrast(self):
        """When 96% of values are >0.8 (the JCBlaise reality), percentile
        binning still produces three meaningfully different bins.

        Without this fix, fixed thresholds 0.6/0.8 put 96% in 'high'."""
        rng = np.random.default_rng(0)
        # 96% in the very-high range (mimicking JCBlaise QLattice output)
        # 4% spread across the rest
        n_high = 9600
        n_other = 400
        high_block = rng.uniform(0.80, 1.00, n_high)
        other_block = rng.uniform(0.20, 0.80, n_other)
        values = np.concatenate([high_block, other_block])
        rng.shuffle(values)

        t_low, t_high = m.compute_percentile_thresholds(values)

        # Apply the bins and count
        bins = {"low": 0, "mid": 0, "high": 0}
        for v in values:
            if v >= t_high:
                bins["high"] += 1
            elif v >= t_low:
                bins["mid"] += 1
            else:
                bins["low"] += 1

        # Each bin should have between 25% and 40% of mass (allow some slop
        # because the distribution is bimodal, but no bin should be <10% or >50%)
        n = len(values)
        for name, count in bins.items():
            frac = count / n
            assert 0.25 <= frac <= 0.40, (
                f"{name}-bin fraction {frac:.2%} outside [25%, 40%]. "
                f"Bins: {bins}, thresholds: low={t_low:.3f}, high={t_high:.3f}"
            )

    def test_compares_against_fixed_threshold_failure_mode(self):
        """Direct demonstration: with 96% high-confidence values, fixed
        thresholds 0.6/0.8 yield massive imbalance; percentile yields balance."""
        rng = np.random.default_rng(0)
        values = np.concatenate([
            rng.uniform(0.80, 1.00, 9600),
            rng.uniform(0.20, 0.80, 400),
        ])
        rng.shuffle(values)

        # FIXED: count using the legacy 0.6/0.8 thresholds
        fixed_high = int((values >= 0.80).sum())
        fixed_mid = int(((values >= 0.60) & (values < 0.80)).sum())
        fixed_low = int((values < 0.60).sum())
        n = len(values)
        # Sanity: fixed should be massively imbalanced
        assert fixed_high / n > 0.85, "expected fixed-threshold high to dominate"

        # PERCENTILE: count using v2.8.6 thresholds
        t_low, t_high = m.compute_percentile_thresholds(values)
        pct_high = int((values >= t_high).sum())
        pct_mid = int(((values >= t_low) & (values < t_high)).sum())
        pct_low = int((values < t_low).sum())

        # Percentile bin counts should be much more balanced than fixed
        max_pct_frac = max(pct_high, pct_mid, pct_low) / n
        max_fixed_frac = max(fixed_high, fixed_mid, fixed_low) / n
        assert max_pct_frac < max_fixed_frac, (
            f"Percentile binning should be more balanced than fixed. "
            f"max_pct={max_pct_frac:.2%}, max_fixed={max_fixed_frac:.2%}"
        )

    def test_handles_nan_values(self):
        values = [0.1, 0.5, np.nan, 0.7, None, 0.9, 0.2, 0.3, 0.6, 0.8, 0.4]
        t_low, t_high = m.compute_percentile_thresholds(values)
        assert t_low < t_high
        assert not np.isnan(t_low)
        assert not np.isnan(t_high)

    def test_falls_back_on_tiny_sample(self):
        """Fewer than 10 valid values → return fallback."""
        t_low, t_high = m.compute_percentile_thresholds(
            [0.5, 0.6, 0.7], fallback=(0.6, 0.8))
        assert (t_low, t_high) == (0.6, 0.8)

    def test_handles_degenerate_distribution(self):
        """All identical values → percentiles collapse, but we spread them."""
        values = [0.7] * 100
        t_low, t_high = m.compute_percentile_thresholds(values)
        # Defensive spread should make t_high strictly greater
        assert t_high > t_low


# compute_percentile_margin_thresholds

class TestComputePercentileMarginThresholds:
    def test_returns_proficient_below_novice(self):
        rng = np.random.default_rng(0)
        margins = rng.uniform(0, 0.5, 200)
        t_proficient, t_novice = m.compute_percentile_margin_thresholds(margins)
        assert t_proficient < t_novice

    def test_jcblaise_like_high_margins_still_balance(self):
        """When most margins are large (clear-cut cases), percentile binning
        should still give roughly 33% advanced/proficient/novice."""
        rng = np.random.default_rng(0)
        # Mostly large margins (clear-cut cases) — JCBlaise reality
        margins = np.concatenate([
            rng.uniform(0.30, 0.50, 9000),  # clear-cut
            rng.uniform(0.00, 0.30, 1000),  # ambiguous
        ])
        rng.shuffle(margins)

        t_proficient, t_novice = m.compute_percentile_margin_thresholds(margins)

        bins = {"advanced": 0, "proficient": 0, "novice": 0}
        for margin in margins:
            if margin >= t_novice:
                bins["novice"] += 1
            elif margin >= t_proficient:
                bins["proficient"] += 1
            else:
                bins["advanced"] += 1

        n = len(margins)
        # No bin should be more than ~40%
        for name, count in bins.items():
            frac = count / n
            assert 0.20 <= frac <= 0.45, (
                f"{name} bin fraction {frac:.2%} outside [20%, 45%]. "
                f"Bins: {bins}"
            )

    def test_falls_back_on_tiny_sample(self):
        t_proficient, t_novice = m.compute_percentile_margin_thresholds(
            [0.1, 0.2], fallback=(0.10, 0.30))
        assert (t_proficient, t_novice) == (0.10, 0.30)


# Static check: the script uses bin_strategy="percentile" by default

class TestScriptDefaults:
    def test_bin_strategy_default_is_percentile(self):
        src = SCRIPT_PATH.read_text(encoding="utf-8")
        assert '"--bin_strategy"' in src
        # Default must be percentile (the v2.8.6 fix)
        assert 'default="percentile"' in src, (
            "Script 10b's default --bin_strategy must be 'percentile' "
            "to fix the JCBlaise control-token imbalance."
        )

    def test_uses_percentile_helpers(self):
        src = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "compute_percentile_thresholds" in src
        assert "compute_percentile_margin_thresholds" in src

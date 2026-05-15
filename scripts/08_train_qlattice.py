#!/usr/bin/env python3
"""Fit a QLattice (feyn) symbolic-regression equation over detector + GNN signals."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

MINERVA_VERSION = "v2.8.4"

FEAT_DIR = Path("data/features")
MODEL_DIR = Path("models")
QL_DIR = MODEL_DIR / "qlattice"
LOG_DIR = Path("logs")

TRAIN = FEAT_DIR / "train_tabular.csv"
VAL = FEAT_DIR / "val_tabular.csv"
TEST = FEAT_DIR / "test_tabular.csv"

OUT_EQ = MODEL_DIR / "qlattice_equation.txt"
OUT_JSON_FLAT = MODEL_DIR / "best_qlattice.json"
OUT_JSON_NESTED = QL_DIR / "best_qlattice.json"
OUT_NOTE = LOG_DIR / "qlattice_notes.txt"

MODEL_DIR.mkdir(exist_ok=True)
QL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# Utilities

def _print_banner() -> None:
    print("=" * 60)
    print(f"  M.I.N.E.R.V.A. {MINERVA_VERSION} — Script 08")
    print(f"  Train QLattice symbolic-regression equation")
    print(f"  Output: {OUT_EQ}")
    print("=" * 60)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _sanitize_name(name: str) -> str:
    """Match minerva_qlattice.py's sanitize_column_name."""
    import re
    out = re.sub(r"[^a-zA-Z0-9]+", "", str(name))
    out = re.sub(r"^([0-9])", r"f\1", out)
    return out


# Tier 1 — feyn QLattice (canonical)

def _try_feyn(train_df: pd.DataFrame, feature_cols: list) -> Optional[str]:
    """Run the canonical feyn QLattice symbolic regression.

    Returns the equation string on success, None on import/runtime failure.
    """
    print("\n[Tier 1] feyn QLattice symbolic regression ...")
    try:
        from feyn import QLattice
    except ImportError as e:
        print(f"  [SKIP] Tier 1: feyn not importable ({e})")
        print(f"         (Wheels exist for Python 3.9-3.12; check your env.)")
        return None
    except Exception as e:
        print(f"  [SKIP] Tier 1: unexpected feyn import error "
              f"({type(e).__name__}: {e})")
        return None

    try:
        data = train_df[feature_cols + ["label"]].copy()
        ql = QLattice()
        print(f"         training on {len(data)} rows x "
              f"{len(feature_cols)} features (this takes ~2-3 min) ...")
        models = ql.auto_run(
            data=data,
            output_name="label",
            n_epochs=20,
        )
        if not models:
            print("  [SKIP] Tier 1: feyn returned zero models")
            return None
        best = models[0]
        eq = str(best.sympify()).strip()
        if not eq:
            print("  [SKIP] Tier 1: feyn returned an empty equation")
            return None
        print(f"  [OK] Tier 1 succeeded — equation length {len(eq)} chars")
        return eq
    except Exception as e:
        print(f"  [SKIP] Tier 1 runtime failure: {type(e).__name__}: {e}")
        return None


# Tier 2 — sklearn LogisticRegression fallback

def build_logreg_equation(train_df: pd.DataFrame, feature_cols: list) -> str:
    """Fit logistic regression and emit a `logreg(...)` equation.

    Returns the equation string. Pure helper — exposed for unit testing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = train_df[feature_cols].astype(float).to_numpy()
    y = train_df["label"].astype(int).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(Xs, y)

    # Build coefficients in original (unscaled) feature space so the
    # equation evaluates correctly against minerva_qlattice's locals.
    # Standardized: z = (x - mu) / sigma
    # Linear:       linear_z = sum(c_i * z_i) + b
    #             = sum(c_i / sigma_i * x_i) - sum(c_i * mu_i / sigma_i) + b
    coefs_z = clf.coef_[0]
    intercept_z = float(clf.intercept_[0])
    sigmas = scaler.scale_
    mus = scaler.mean_

    coefs_x = coefs_z / sigmas
    intercept_x = intercept_z - float((coefs_z * mus / sigmas).sum())

    # Format: logreg(intercept + c1*sanitized_name1 + c2*sanitized_name2 + ...)
    parts = [f"{intercept_x:.6g}"]
    for c, name in zip(coefs_x, feature_cols):
        san = _sanitize_name(name)
        if c >= 0:
            parts.append(f" + {c:.6g}*{san}")
        else:
            parts.append(f" - {abs(c):.6g}*{san}")
    return "logreg(" + "".join(parts) + ")"


def _logreg_fallback(train_df: pd.DataFrame, feature_cols: list) -> str:
    """Fit logistic regression and verify the equation evaluates correctly."""
    print("\n[Tier 2] sklearn LogisticRegression fallback ...")
    eq = build_logreg_equation(train_df, feature_cols)

    # Sanity-check: can the equation be compiled/evaluated by the evaluator?
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import minerva_qlattice as ml
        code = ml.compile_equation(eq)
        bundle = ml.build_feature_locals(train_df[feature_cols])
        out = ml.eval_compiled(code, bundle.locals, n_rows=len(train_df))
        print(f"  [OK] Tier 2 equation compiles + evaluates "
              f"(mean prediction {float(out.mean()):.3f})")
    except Exception as e:
        # Self-check failed — surface it. The pipeline will hit it later otherwise.
        raise RuntimeError(
            f"Tier 2 fallback produced an equation that the QLattice "
            f"evaluator cannot parse: {type(e).__name__}: {e}\n"
            f"Equation: {eq}"
        )

    print(f"  [OK] Tier 2 succeeded — equation length {len(eq)} chars")
    return eq


# Main

def main() -> None:
    _print_banner()

    for p in [TRAIN, VAL, TEST]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run 06_extract_features.py first."
            )

    train_df = pd.read_csv(TRAIN)
    val_df = pd.read_csv(VAL)
    test_df = pd.read_csv(TEST)

    feature_cols = [
        c for c in train_df.columns
        if c not in {"id", "label", "dataset", "lang", "split"}
    ]
    if not feature_cols:
        raise RuntimeError("No usable feature columns found.")

    print(f"\nFeatures: {len(feature_cols)} columns "
          f"(sample: {feature_cols[:5]}{'...' if len(feature_cols) > 5 else ''})")
    print(f"Splits: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    eq: Optional[str] = None
    source: Optional[str] = None

    # Tier 1 — feyn (canonical)
    eq = _try_feyn(train_df, feature_cols)
    if eq is not None:
        source = "feyn (QLattice)"

    # Tier 2 — LogisticRegression fallback
    if eq is None:
        eq = _logreg_fallback(train_df, feature_cols)
        source = "sklearn LogisticRegression (fallback)"

    if not eq:
        raise RuntimeError("All tiers failed to produce an equation.")

    # Persist
    OUT_EQ.write_text(eq, encoding="utf-8")

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "minerva_version": MINERVA_VERSION,
        "source": source,
        "equation": eq,
        "feature_cols": feature_cols,
        "train_file": str(TRAIN),
        "val_file": str(VAL),
        "test_file": str(TEST),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "notes": [
            "Plain-text equation goes to models/qlattice_equation.txt",
            "JSON metadata records source tier (feyn vs fallback) for reproducibility.",
            "Downstream scripts (13, 18, 10b) consume the .txt file unchanged.",
        ],
    }
    _write_json(OUT_JSON_FLAT, payload)
    _write_json(OUT_JSON_NESTED, payload)

    print(f"\n[OK] Equation saved to: {OUT_EQ}")
    print(f"[OK] Metadata saved to: {OUT_JSON_FLAT}")
    print(f"[OK] Source: {source}")
    print(f"\nEquation:")
    if len(eq) > 240:
        print(f"  {eq[:240]}...")
    else:
        print(f"  {eq}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("\n[FATAL] Unhandled exception:")
        traceback.print_exc()
        sys.exit(1)

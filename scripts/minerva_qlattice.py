from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Set

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Safe math


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid for numpy arrays."""
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


SAFE_FUNCS: Dict[str, Any] = {
    # common qlattice ops
    "logreg": _sigmoid,
    "sigmoid": _sigmoid,
    "exp": np.exp,
    "log": np.log,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
    "sin": np.sin,
    "cos": np.cos,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "min": np.minimum,
    "max": np.maximum,
    "where": np.where,
    "clip": np.clip,
    # constants (occasionally appear)
    "pi": float(np.pi),
    "e": float(np.e),
}


# -----------------------------------------------------------------------------
# Name handling


def sanitize_column_name(name: str) -> str:
    """Sanitize a feature name to a simple alphanumeric identifier.

    Mirrors common Qlattice/Feyn name emission where non-alphanumerics are
    removed: e.g. `r_pca_0` -> `rpca0`.
    """

    out = re.sub(r"[^a-zA-Z0-9]+", "", str(name))
    # ensure valid python identifier
    out = re.sub(r"^([0-9])", r"f\1", out)
    return out


def extract_variable_names(expr: str) -> Set[str]:
    """Extract identifier-like tokens referenced by the equation."""

    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    return {t for t in tokens if t not in SAFE_FUNCS}


@dataclass
class LocalsBundle:
    locals: Dict[str, Any]
    alias_to_source: Dict[str, str]


def build_feature_locals(df: pd.DataFrame) -> LocalsBundle:
    """Build a locals dict for eval() exposing original + sanitized feature names."""

    local_vars: Dict[str, Any] = {}
    alias_to_source: Dict[str, str] = {}

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        arr = df[c].to_numpy(dtype=float)

        # original name
        if c not in local_vars:
            local_vars[c] = arr
            alias_to_source[c] = c

        # sanitized alias
        s = sanitize_column_name(c)
        if s and s not in local_vars:
            local_vars[s] = arr
            alias_to_source[s] = c

    # functions override any collisions
    local_vars.update(SAFE_FUNCS)

    return LocalsBundle(locals=local_vars, alias_to_source=alias_to_source)


# -----------------------------------------------------------------------------
# Evaluation


def compile_equation(expr: str):
    expr_py = str(expr).strip().replace("^", "**")
    if not expr_py:
        raise ValueError("Equation is empty")
    return compile(expr_py, "<qlattice_equation>", "eval")


def eval_compiled(code, local_vars: Mapping[str, Any], n_rows: int) -> np.ndarray:
    out = eval(code, {"__builtins__": {}}, dict(local_vars))
    out = np.asarray(out, dtype=float)
    if out.ndim == 0:
        out = np.full((n_rows,), float(out), dtype=float)
    return out

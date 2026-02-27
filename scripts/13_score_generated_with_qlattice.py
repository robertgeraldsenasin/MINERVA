import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically-stable sigmoid for arrays.
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


SAFE_FUNCS: Dict[str, object] = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
    "sin": np.sin,
    "cos": np.cos,
    "abs": np.abs,
    # Feyn classification models often wrap equations in logreg(...)
    "logreg": _sigmoid,
    "sigmoid": _sigmoid,
}


def sanitize_column_name(name: str) -> str:
    # Keep only alphanumerics to match how equation terms are usually emitted (e.g., r_pca_0 -> rpca0).
    out = re.sub(r"[^a-zA-Z0-9]+", "", name)
    out = re.sub(r"^([0-9])", r"f\1", out)  # ensure valid python identifiers
    return out


def build_sanitized_view(df: pd.DataFrame) -> pd.DataFrame:
    sanitized = pd.DataFrame(index=df.index)
    for col in df.columns:
        sanitized[sanitize_column_name(col)] = df[col]
    return sanitized


def extract_variable_names(expr: str) -> Set[str]:
    # Extract identifier-like tokens, excluding known safe functions.
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    return {t for t in tokens if t not in SAFE_FUNCS}


def eval_equation(expr: str, df_sanitized: pd.DataFrame) -> np.ndarray:
    local_vars = {
        col: df_sanitized[col].values for col in df_sanitized.columns}
    # Ensure function names win if there is any collision
    local_vars.update(SAFE_FUNCS)
    try:
        out = eval(expr, {"__builtins__": {}}, local_vars)
    except Exception as e:
        missing = sorted(list(extract_variable_names(
            expr) - set(df_sanitized.columns)))
        raise RuntimeError(
            "Failed to evaluate Qlattice equation.\n"
            f"Error: {repr(e)}\n"
            f"Missing variables (if any): {missing}\n"
            f"Equation: {expr}"
        )
    out = np.asarray(out, dtype=float)
    return out


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_jsonl", default="generated/gpt2_synthetic_samples.jsonl")
    ap.add_argument("--equation", default="models/qlattice_equation.txt")
    ap.add_argument("--target", default="fake", choices=["fake", "real"])
    ap.add_argument("--min_margin", type=float, default=0.05)
    ap.add_argument(
        "--out_scored", default="generated/gpt2_synthetic_scored.jsonl")
    ap.add_argument(
        "--out_final", default="generated/gpt2_synthetic_final.jsonl")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing generated file: {in_path}\n"
            f"Fix: run scripts/12_generate_gpt2MINERVA.py first."
        )

    eq_path = Path(args.equation)
    if not eq_path.exists():
        raise FileNotFoundError(
            f"Missing Qlattice equation file: {eq_path}\n"
            f"Fix: run scripts/08_train_qlattice.py first."
        )

    rows = read_jsonl(in_path)
    if not rows:
        raise RuntimeError(f"Input JSONL is empty: {in_path}")

    df = pd.DataFrame(rows)
    eq = eq_path.read_text(encoding="utf-8").strip()
    if not eq:
        raise RuntimeError(f"Equation file is empty: {eq_path}")

    # Build sanitized view for both (a) missing-variable checks and (b) evaluation.
    sdf = build_sanitized_view(df)

    needed = extract_variable_names(eq)
    missing = sorted(list(needed - set(sdf.columns)))
    if missing:
        raise RuntimeError(
            "Input JSONL is missing feature columns required by the Qlattice equation (after sanitization).\n"
            f"Missing: {missing}\n\n"
            "Notes:\n"
            " - Qlattice equations typically use names like rpca0/dpca10 (underscores removed).\n"
            " - Your JSONL should contain the corresponding r_pca_* / d_pca_* columns.\n\n"
            "Fix:\n"
            " - Ensure PCA files exist: models/pca_roberta.joblib and models/pca_distilbert.joblib\n"
            " - Re-run scripts/12_generate_gpt2MINERVA.py so it outputs PCA columns\n"
            " - Or retrain Qlattice on a feature set that matches your generated JSONL."
        )

    scores = eval_equation(eq, sdf)
    if scores.shape[0] != len(df):
        raise RuntimeError(
            f"Equation output has wrong length: got {scores.shape[0]} expected {len(df)}")

    # Interpret as probability of FAKE (typical when equation uses logreg()).
    p_fake = np.clip(scores, 0.0, 1.0)

    df["p_qlattice_fake"] = p_fake.astype(float)
    df["score_margin"] = np.abs(df["p_qlattice_fake"] - 0.5).astype(float)

    # Write scored JSONL (all rows)
    scored_rows = df.to_dict(orient="records")
    write_jsonl(Path(args.out_scored), scored_rows)
    print(f"[13] Scored -> {args.out_scored} (rows={len(scored_rows)})")

    # Filter to target + margin
    keep = df[df.get("target", args.target) == args.target].copy(
    ) if "target" in df.columns else df.copy()
    keep = keep[keep["score_margin"] >= args.min_margin].copy()
    final_rows = keep.to_dict(orient="records")
    write_jsonl(Path(args.out_final), final_rows)
    print(f"[13] Final -> {args.out_final} (rows={len(final_rows)})")


if __name__ == "__main__":
    main()

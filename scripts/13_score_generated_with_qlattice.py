from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from minerva_qlattice import build_feature_locals, compile_equation, eval_compiled, extract_variable_names


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def difficulty_bin_from_margin(m: float) -> str:
    # Margin here is |p(fake) - 0.5|. Larger => easier.
    if m >= 0.25:
        return "easy"
    if m >= 0.15:
        return "medium"
    return "hard"


def _infer_target_col(df: pd.DataFrame) -> str | None:
    for c in ["target", "target_label"]:
        if c in df.columns:
            return c
    return None


def main() -> None:
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
            f"Missing generated file: {in_path}\nFix: run scripts/12_generate_gpt2MINERVA.py first."
        )

    eq_path = Path(args.equation)
    if not eq_path.exists():
        raise FileNotFoundError(
            f"Missing Qlattice equation file: {eq_path}\nFix: run scripts/08_train_qlattice.py first."
        )

    rows = read_jsonl(in_path)
    if not rows:
        raise RuntimeError(f"Input JSONL is empty: {in_path}")

    df = pd.DataFrame(rows)

    eq = eq_path.read_text(encoding="utf-8").strip()
    if not eq:
        raise RuntimeError(f"Equation file is empty: {eq_path}")

    code = compile_equation(eq)
    bundle = build_feature_locals(df)
    needed = extract_variable_names(eq)
    missing = sorted(list(needed - set(bundle.locals.keys())))
    if missing:
        raise RuntimeError(
            "Input JSONL is missing feature columns required by the Qlattice equation.\n"
            f"Missing: {missing}\n\n"
            "Fix:\n"
            " - Ensure Script 12 outputs PCA columns if your equation uses r_pca_* / d_pca_*\n"
            " - Ensure lexical feature names match Script 06 (char_len, word_len, exclam, question, digit_ratio)\n"
            " - Or retrain Qlattice on a feature set that matches your generated JSONL."
        )

    scores = eval_compiled(code, bundle.locals, n_rows=len(df))
    if scores.shape[0] != len(df):
        raise RuntimeError(
            f"Equation output has wrong length: got {scores.shape[0]} expected {len(df)}")

    # Interpret equation output as probability of FAKE (common when equation uses logreg()).
    p_fake = np.clip(scores.astype(float), 0.0, 1.0)

    df["qlattice_equation_source"] = str(eq_path)
    df["qlattice_threshold"] = 0.5
    df["qlattice_direction"] = ">="
    df["qlattice_score"] = scores.astype(float)
    df["p_qlattice_fake"] = p_fake.astype(float)
    df["qlattice_margin"] = (df["p_qlattice_fake"] - 0.5).astype(float)
    df["qlattice_pred"] = (df["p_qlattice_fake"] >= 0.5).astype(int)
    df["score_margin"] = np.abs(df["p_qlattice_fake"] - 0.5).astype(float)
    df["difficulty_bin"] = df["score_margin"].apply(
        lambda x: difficulty_bin_from_margin(float(x)))

    scored_rows = df.to_dict(orient="records")
    write_jsonl(Path(args.out_scored), scored_rows)
    print(f"[13] Scored -> {args.out_scored} (rows={len(scored_rows)})")

    # Filter to target + margin
    target_col = _infer_target_col(df)
    keep = df.copy()
    if target_col is not None:
        keep = keep[keep[target_col] == args.target].copy()
    keep = keep[keep["score_margin"] >= float(args.min_margin)].copy()

    final_rows = keep.to_dict(orient="records")
    write_jsonl(Path(args.out_final), final_rows)
    print(f"[13] Final  -> {args.out_final} (rows={len(final_rows)})")


if __name__ == "__main__":
    main()

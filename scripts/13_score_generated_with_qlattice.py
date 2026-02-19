from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


SAFE_FUNCS = {
    "log": np.log,
    "log1p": np.log1p,
    "exp": np.exp,
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
    "np": np,
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_variable_names(expr: str) -> set[str]:
    tokens = set(re.findall(r"[A-Za-z_]\w*", expr))
    tokens = {t for t in tokens if t not in SAFE_FUNCS}
    tokens -= {"e", "pi", "True", "False"}
    return tokens


def eval_equation(expr: str, df: pd.DataFrame) -> np.ndarray:
    expr = expr.strip().replace("^", "**")
    local_vars = {c: df[c].to_numpy(
        dtype=float) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    local_vars.update(SAFE_FUNCS)

    try:
        out = eval(expr, {"__builtins__": {}}, local_vars)
    except Exception as e:
        missing = extract_variable_names(expr) - set(df.columns)
        raise RuntimeError(
            f"Failed to evaluate Qlattice equation.\n"
            f"Error: {repr(e)}\n"
            f"Missing variables (if any): {sorted(list(missing))}\n"
            f"Equation: {expr}"
        )

    out = np.asarray(out, dtype=float)
    if out.ndim == 0:
        out = np.full((len(df),), float(out))
    return out


def calibrate_threshold(scores: np.ndarray, y_true: np.ndarray) -> tuple[float, str, float]:
    mask = np.isfinite(scores)
    scores = scores[mask]
    y_true = y_true[mask].astype(int)

    if len(scores) == 0:
        return 0.5, ">=", 0.0

    qs = np.quantile(scores, np.linspace(0.01, 0.99, 199))

    best_thr = float(qs[0])
    best_dir = ">="
    best_f1 = -1.0

    for thr in qs:
        pred_ge = (scores >= thr).astype(int)
        f1_ge = f1_score(y_true, pred_ge, zero_division=0)

        pred_le = (scores <= thr).astype(int)
        f1_le = f1_score(y_true, pred_le, zero_division=0)

        if f1_ge > best_f1:
            best_f1 = float(f1_ge)
            best_thr = float(thr)
            best_dir = ">="
        if f1_le > best_f1:
            best_f1 = float(f1_le)
            best_thr = float(thr)
            best_dir = "<="

    return best_thr, best_dir, best_f1


def difficulty_bins(margins: np.ndarray, k: int = 3) -> list[str]:
    if k <= 1:
        return ["easy"] * len(margins)

    m = np.asarray(margins, dtype=float)
    if np.allclose(m, m[0]):
        return ["medium"] * len(m)

    cuts = np.quantile(m, np.linspace(0, 1, k + 1))
    cuts = np.unique(cuts)

    labels = []
    for v in m:
        idx = int(np.searchsorted(cuts, v, side="right") - 1)
        idx = max(0, min(idx, len(cuts) - 2))

        if idx == 0:
            labels.append("hard")
        elif idx == (len(cuts) - 2):
            labels.append("easy")
        else:
            labels.append("medium")
    return labels


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MINERVA Script 13: Score generated samples with Qlattice equation and export simulator-ready subsets.")

    ap.add_argument(
        "--in_file", default="generated/gpt2_synthetic_samples.jsonl")
    ap.add_argument("--equation", default="models/qlattice_equation.txt")
    ap.add_argument("--calib_val", default="data/features/val_tabular.csv")

    ap.add_argument(
        "--out_scored", default="generated/gpt2_synthetic_scored.jsonl")
    ap.add_argument(
        "--out_final", default="generated/gpt2_synthetic_final.jsonl")

    ap.add_argument("--target", choices=["fake", "real", "any"], default="any",
                    help="Filter final output by target_label field in the JSONL (if present).")
    ap.add_argument("--min_margin", type=float, default=0.0,
                    help="Minimum margin from Qlattice boundary for inclusion in final set.")
    ap.add_argument("--difficulty_k", type=int, default=3,
                    help="Number of difficulty bins (default 3 => hard/medium/easy).")

    ap.add_argument("--no_calibrate", action="store_true",
                    help="If set, skip calibration and use threshold=0.5, direction='>='.")

    args = ap.parse_args()

    in_path = Path(args.in_file)
    eq_path = Path(args.equation)
    val_path = Path(args.calib_val)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run script 12 first.")
    if not eq_path.exists():
        raise FileNotFoundError(
            f"Missing {eq_path}. Run 08_train_qlattice.py first.")

    eq = eq_path.read_text(encoding="utf-8").strip()
    if not eq:
        raise RuntimeError(f"Empty equation file: {eq_path}")

    rows = read_jsonl(in_path)
    df = pd.DataFrame(rows)

    needed_vars = extract_variable_names(eq)
    missing = needed_vars - set(df.columns)
    if missing:
        raise RuntimeError(
            "Generated JSONL is missing feature columns required by the Qlattice equation.\n"
            f"Missing: {sorted(list(missing))}\n\n"
            "Fix: re-run script 12 with PCA files present (models/pca_roberta.joblib and models/pca_distilbert.joblib),\n"
            "or retrain Qlattice on a smaller feature set that matches what script 12 outputs."
        )

    scores = eval_equation(eq, df)
    df["qlattice_score"] = scores

    thr = 0.5
    direction = ">="
    best_f1 = None

    if not args.no_calibrate and val_path.exists():
        val_df = pd.read_csv(val_path)
        missing_val = needed_vars - set(val_df.columns)
        if missing_val:
            print(
                "[WARN] Val tabular missing columns required by equation; skipping calibration.")
        else:
            val_scores = eval_equation(eq, val_df)
            y_val = val_df["label"].to_numpy(dtype=int)
            thr, direction, best_f1 = calibrate_threshold(val_scores, y_val)

            if direction == ">=":
                pred = (val_scores >= thr).astype(int)
                margins = val_scores - thr
            else:
                pred = (val_scores <= thr).astype(int)
                margins = thr - val_scores

            acc = accuracy_score(y_val, pred)
            p, r, f1, _ = precision_recall_fscore_support(
                y_val, pred, average="binary", zero_division=0)
            print(
                f"[13][CALIB] thr={thr:.6f} dir={direction} acc={acc:.4f} p={p:.4f} r={r:.4f} f1={f1:.4f}")

    df["qlattice_threshold"] = float(thr)
    df["qlattice_direction"] = direction

    if direction == ">=":
        df["qlattice_pred"] = (df["qlattice_score"] >= thr).astype(int)
        df["qlattice_margin"] = (df["qlattice_score"] - thr).astype(float)
    else:
        df["qlattice_pred"] = (df["qlattice_score"] <= thr).astype(int)
        df["qlattice_margin"] = (thr - df["qlattice_score"]).astype(float)

    df["difficulty_bin"] = difficulty_bins(
        df["qlattice_margin"].to_numpy(), k=args.difficulty_k)

    out_scored = Path(args.out_scored)
    write_jsonl(out_scored, df.to_dict(orient="records"))
    print(f"[13] Scored -> {out_scored.resolve()} (rows={len(df)})")

    final_df = df.copy()

    if args.target != "any" and "target_label" in final_df.columns:
        final_df = final_df[final_df["target_label"] == args.target]

    if args.target == "fake":
        final_df = final_df[final_df["qlattice_pred"] == 1]
    elif args.target == "real":
        final_df = final_df[final_df["qlattice_pred"] == 0]

    if args.min_margin > 0:
        final_df = final_df[final_df["qlattice_margin"] >= args.min_margin]

    out_final = Path(args.out_final)
    write_jsonl(out_final, final_df.to_dict(orient="records"))
    print(f"[13] Final -> {out_final.resolve()} (rows={len(final_df)})")


if __name__ == "__main__":
    main()

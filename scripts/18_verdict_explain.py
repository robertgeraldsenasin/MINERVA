from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Qlattice equation evaluation (safe subset)


def logreg(x):
    """Logistic/sigmoid function used by Qlattice `logreg(·)` equations.

    Qlattice sometimes wraps a linear expression with `logreg(...)` to
    constrain outputs to (0,1). We implement it as a numerically-stable
    sigmoid.
    """
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


SAFE_FUNCS = {
    "logreg": logreg,
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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def extract_variable_names(expr: str) -> List[str]:
    """Extract candidate feature names referenced by the equation."""
    tokens = set(re.findall(r"[A-Za-z_]\w*", expr))
    tokens = {t for t in tokens if t not in SAFE_FUNCS}
    tokens -= {"e", "pi", "True", "False"}
    return sorted(tokens)


def eval_equation(expr: str, df: pd.DataFrame) -> np.ndarray:
    """Evaluate the equation using numeric columns in df."""
    expr = expr.strip().replace("^", "**")
    local_vars = {
        c: df[c].to_numpy(dtype=float)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    }
    local_vars.update(SAFE_FUNCS)
    try:
        out = eval(expr, {"__builtins__": {}}, local_vars)
    except Exception as e:
        missing = set(extract_variable_names(expr)) - set(df.columns)
        raise RuntimeError(
            "Failed to evaluate Qlattice equation.\n"
            f"Error: {repr(e)}\n"
            f"Missing variables (if any): {sorted(list(missing))}\n"
            f"Equation: {expr}"
        )
    out = np.asarray(out, dtype=float)
    if out.ndim == 0:
        out = np.full((len(df),), float(out))
    return out


# -----------------------------------------------------------------------------
# Explanation helpers

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def float_or_none(x: Any) -> float | None:
    """Convert to float, but return None for missing / non-finite values.

    This keeps the exported JSON strictly valid (no NaN/Infinity).
    """
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def basic_heuristics(text: str) -> Dict[str, float]:
    if not isinstance(text, str):
        text = ""
    urls = len(URL_RE.findall(text))
    excls = text.count("!")
    qmarks = text.count("?")
    letters = [c for c in text if c.isalpha()]
    caps = [c for c in letters if c.isupper()]
    caps_ratio = (len(caps) / len(letters)) if letters else 0.0
    return {
        "url_count": float(urls),
        "exclamation_count": float(excls),
        "question_count": float(qmarks),
        "caps_ratio": float(caps_ratio),
    }


def humanize_feature(name: str) -> str:
    """Map feature names to human‑readable labels for the game UI."""
    if name == "p_roberta_fake":
        return "RoBERTa fake probability"
    if name == "p_distil_fake":
        return "DistilBERT fake probability"
    if name.startswith("r_pca_"):
        return f"RoBERTa semantic component {name.split('_')[-1]}"
    if name.startswith("d_pca_"):
        return f"DistilBERT semantic component {name.split('_')[-1]}"
    if name == "exclam":
        return "Exclamation marks"
    if name == "question":
        return "Question marks"
    if name == "digit_ratio":
        return "Digit ratio"
    if name == "char_len":
        return "Character length"
    if name == "word_len":
        return "Word count"
    return name


@dataclass
class VerdictRecord:
    id: str
    target_label: str | None
    text: str
    verdict: str
    fake_likelihood_percent: float
    credibility_percent: float
    difficulty_bin: str
    metadata: Dict[str, Any]
    qlattice: Dict[str, Any]
    detectors: Dict[str, Any]
    heuristics: Dict[str, Any]
    explanation: Dict[str, Any]


def choose_default_infile() -> Path:
    """Pick the best available default input file."""
    candidates = [
        Path("generated/gpt2_synthetic_final.jsonl"),
        Path("generated/gpt2_synthetic_scored.jsonl"),
        Path("generated/gpt2_synthetic_samples.jsonl"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def compute_margin(score: float, thr: float, direction: str) -> float:
    if direction == ">=":
        return float(score - thr)
    return float(thr - score)


def local_contributions(
    expr: str,
    variables: List[str],
    row: Dict[str, Any],
    baseline: Dict[str, float],
    thr: float,
    direction: str,
) -> List[Dict[str, Any]]:
    """Counterfactual per-feature contributions to Qlattice margin.

    For each variable v:
      delta = margin(original) - margin(row with v := baseline[v])

    Interpretation:
      delta > 0  => v pushes toward FAKE (increases margin)
      delta < 0  => v pushes toward REAL (decreases margin)
    """

    df0 = pd.DataFrame([row])
    s0 = float(eval_equation(expr, df0)[0])
    m0 = compute_margin(s0, thr, direction)

    out: List[Dict[str, Any]] = []
    for v in variables:
        if v not in baseline:
            continue
        cf = dict(row)
        cf[v] = float(baseline[v])
        s1 = float(eval_equation(expr, pd.DataFrame([cf]))[0])
        m1 = compute_margin(s1, thr, direction)
        delta = float(m0 - m1)
        out.append(
            {
                "feature": v,
                "feature_human": humanize_feature(v),
                "value": float_or_none(row.get(v)),
                "baseline": float(baseline[v]),
                "delta_margin": delta,
                "direction": "push_fake" if delta > 0 else "push_real" if delta < 0 else "neutral",
            }
        )

    out.sort(key=lambda r: abs(r["delta_margin"]), reverse=True)
    return out


def build_explanation_text(
    verdict: str,
    fake_pct: float,
    top_factors: List[Dict[str, Any]],
    heur: Dict[str, float],
) -> Dict[str, Any]:
    """Create user-facing explanation strings (safe educational framing)."""

    # Top 3 “push toward verdict” factors
    push_fake = [f for f in top_factors if f["direction"] == "push_fake"]
    push_real = [f for f in top_factors if f["direction"] == "push_real"]

    if verdict == "fake":
        key = push_fake[:3]
    else:
        key = push_real[:3]

    signals: List[str] = []
    for f in key:
        # Keep the message informative but not “how-to-evade”
        signals.append(
            f"{f['feature_human']} is unusually {'high' if f['direction']=='push_fake' else 'low'} "
            f"relative to baseline."
        )

    # Add simple heuristic flags
    if heur.get("url_count", 0) >= 1:
        signals.append(
            "Contains link(s): verify the source domain and the original publisher.")
    if heur.get("exclamation_count", 0) >= 3:
        signals.append(
            "Heavy use of exclamation marks can indicate sensational or persuasive framing.")
    if heur.get("caps_ratio", 0) >= 0.2:
        signals.append(
            "High ALL‑CAPS ratio can indicate emotionally charged messaging.")

    if not signals:
        signals.append(
            "The equation-based score was close to the decision boundary; treat as uncertain and verify carefully.")

    summary = (
        f"Verdict: {verdict.upper()} (estimated fake-likelihood {fake_pct:.1f}%). "
        "This decision is primarily based on the interpretable equation (Qlattice) applied to detector and lexical features."
    )

    why_it_matters = (
        "Misinformation can influence opinions and decisions. "
        "Before sharing, cross-check with reliable sources and look for corroboration."
    )

    what_to_do = [
        "Check the original source (who published it, when, and where).",
        "Look for the same claim reported by multiple credible outlets.",
        "Be cautious with highly emotional language or urgent calls to share.",
    ]

    return {
        "summary": summary,
        "signals": signals,
        "why_it_matters": why_it_matters,
        "what_to_do": what_to_do,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "MINERVA Script 18: Create verdict/explanation JSON for Unity using the Qlattice equation. "
            "(Run with no args after scripts 10–13.)"
        )
    )

    ap.add_argument(
        "--in_file",
        default=None,
        help="Input JSONL (default: generated/gpt2_synthetic_final.jsonl if present).",
    )
    ap.add_argument(
        "--out_file",
        default="generated/gpt2_synthetic_verdicts.json",
        help="Output verdict file (default: generated/gpt2_synthetic_verdicts.json).",
    )
    ap.add_argument(
        "--out_format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format (json array is Unity-friendly; jsonl is stream-friendly).",
    )
    ap.add_argument(
        "--equation",
        default="models/qlattice_equation.txt",
        help="Path to Qlattice equation (default: models/qlattice_equation.txt).",
    )
    ap.add_argument(
        "--calib_val",
        default="data/features/val_tabular.csv",
        help="Validation tabular CSV for baselines/calibration (default: data/features/val_tabular.csv).",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many top contributing factors to store per record.",
    )
    ap.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Optional cap for debugging (process only first N records).",
    )
    ap.add_argument(
        "--include_equation",
        action="store_true",
        help="If set, include the equation string in every record (bigger output).",
    )

    args = ap.parse_args()

    in_path = Path(args.in_file) if args.in_file else choose_default_infile()
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {in_path}\n"
            "Expected after running scripts 10→13.\n"
            "Try: python scripts/10_prepare_gpt2MINERVA.py && python scripts/11_train_gpt2MINERVA.py && "
            "python scripts/12_generate_gpt2MINERVA.py ... && python scripts/13_score_generated_with_qlattice.py"
        )

    eq_path = Path(args.equation)
    if not eq_path.exists():
        raise FileNotFoundError(
            f"Missing equation file: {eq_path}\n"
            "Run: python scripts/08_train_qlattice.py"
        )

    equation = eq_path.read_text(encoding="utf-8").strip()
    if not equation:
        raise RuntimeError(f"Empty equation file: {eq_path}")

    rows = read_jsonl(in_path)
    if args.max_records:
        rows = rows[: int(args.max_records)]

    if not rows:
        raise RuntimeError(f"Input file is empty: {in_path}")

    df = pd.DataFrame(rows)

    variables = extract_variable_names(equation)
    missing = set(variables) - set(df.columns)
    if missing:
        raise RuntimeError(
            "Input JSONL is missing feature columns required by the Qlattice equation.\n"
            f"Missing: {sorted(list(missing))}\n\n"
            "Fix: re-run script 12 (ensure PCA files exist if your equation uses r_pca_* / d_pca_*), "
            "then re-run script 13."
        )

    # Determine threshold/direction (prefer what script 13 wrote)
    thr = float(df["qlattice_threshold"].iloc[0]
                ) if "qlattice_threshold" in df.columns else 0.5
    direction = str(df["qlattice_direction"].iloc[0]
                    ) if "qlattice_direction" in df.columns else ">="
    if direction not in (">=", "<="):
        direction = ">="

    # Baseline values for local contributions + scale for mapping margin→probability
    baseline: Dict[str, float] = {}
    scale = 1.0
    val_path = Path(args.calib_val)
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        if set(variables).issubset(set(val_df.columns)):
            baseline = {v: float(val_df[v].median()) for v in variables}
            # Use val margins to set a robust sigmoid scale
            val_scores = eval_equation(equation, val_df)
            val_margins = np.array([compute_margin(s, thr, direction)
                                   for s in val_scores], dtype=float)
            q = float(np.quantile(np.abs(val_margins), 0.75))
            scale = q if q > 1e-9 else float(np.std(val_margins) + 1e-6)
        else:
            # fall back to generated set
            baseline = {v: float(df[v].median()) for v in variables}
    else:
        baseline = {v: float(df[v].median()) for v in variables}

    # If the input already has qlattice_score, reuse it; else compute.
    if "qlattice_score" in df.columns and pd.api.types.is_numeric_dtype(df["qlattice_score"]):
        scores = df["qlattice_score"].to_numpy(dtype=float)
    else:
        scores = eval_equation(equation, df)
        df["qlattice_score"] = scores

    # Ensure margin/pred are available
    margins = np.array([compute_margin(s, thr, direction)
                       for s in scores], dtype=float)
    df["qlattice_margin"] = margins
    df["qlattice_pred"] = (margins >= 0).astype(int)

    # Difficulty bin may already exist from script 13
    if "difficulty_bin" not in df.columns:
        # simple fallback: split into 3 quantiles of margin
        qs = np.quantile(margins, [0.33, 0.66])
        bins = []
        for m in margins:
            if m <= qs[0]:
                bins.append("hard")
            elif m <= qs[1]:
                bins.append("medium")
            else:
                bins.append("easy")
        df["difficulty_bin"] = bins

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(df.to_dict(orient="records")):
        rid = str(row.get("id", f"row_{i}"))
        target_label = row.get("target_label")
        text = str(row.get("text", ""))

        score = float(row.get("qlattice_score"))
        margin = float(row.get("qlattice_margin"))
        pred = int(row.get("qlattice_pred"))
        verdict = "fake" if pred == 1 else "real"

        # Map margin to a probability-like number using a robust sigmoid
        p_fake = float(sigmoid(margin / max(1e-6, scale)))
        fake_pct = float(100.0 * p_fake)
        cred_pct = float(100.0 * (1.0 - p_fake))

        # Compute per-feature contributions to the margin (Qlattice is the core explainer)
        contribs = local_contributions(
            equation,
            variables=variables,
            row=row,
            baseline=baseline,
            thr=thr,
            direction=direction,
        )
        if int(args.top_k) == 0:
            top_factors = contribs
        else:
            top_factors = contribs[: max(1, int(args.top_k))]

        heur = basic_heuristics(text)

        detectors: Dict[str, Any] = {
            "p_roberta_fake": float_or_none(row.get("p_roberta_fake")),
            "p_distil_fake": float_or_none(row.get("p_distil_fake")),
        }
        pr = detectors.get("p_roberta_fake")
        pd_ = detectors.get("p_distil_fake")
        if pr is not None and pd_ is not None:
            detectors["p_ensemble_fake"] = float((pr + pd_) / 2.0)

        metadata = {
            "accept_mode": row.get("accept_mode"),
            "min_conf": float_or_none(row.get("min_conf")),
            "temperature": float_or_none(row.get("temperature")),
            "top_p": float_or_none(row.get("top_p")),
            "seed": row.get("seed"),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        explanation = build_explanation_text(
            verdict=verdict,
            fake_pct=fake_pct,
            top_factors=top_factors,
            heur=heur,
        )

        rec = VerdictRecord(
            id=rid,
            target_label=str(
                target_label) if target_label is not None else None,
            text=text,
            verdict=verdict,
            fake_likelihood_percent=fake_pct,
            credibility_percent=cred_pct,
            difficulty_bin=str(row.get("difficulty_bin", "medium")),
            metadata=metadata,
            qlattice={
                "score": score,
                "threshold": float(thr),
                "direction": direction,
                "margin": margin,
                "pred": pred,
                "top_factors": top_factors,
                **({"equation": equation} if args.include_equation else {}),
            },
            detectors=detectors,
            heuristics=heur,
            explanation=explanation,
        )
        out_rows.append(asdict(rec))

    out_path = Path(args.out_file)
    if args.out_format == "jsonl":
        write_jsonl(out_path, out_rows)
    else:
        write_json(out_path, out_rows)

    print(f"[18] Input : {in_path.resolve()}")
    print(f"[18] Output: {out_path.resolve()} (rows={len(out_rows)})")


if __name__ == "__main__":
    main()

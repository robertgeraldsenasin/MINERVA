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

from minerva_qlattice import (
    SAFE_FUNCS,
    build_feature_locals,
    compile_equation,
    eval_compiled,
    extract_variable_names,
)


# -----------------------------------------------------------------------------
# IO helpers


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


# -----------------------------------------------------------------------------
# Explanation helpers


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def float_or_none(x: Any) -> float | None:
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
    # Support both original and sanitized aliases.
    if name in {"p_roberta_fake", "probertafake"}:
        return "RoBERTa fake probability"
    if name in {"p_distil_fake", "pdistilfake"}:
        return "DistilBERT fake probability"
    if name in {"p_degnn_fake", "pdegnnfake"}:
        return "DE-GNN fake probability"

    m = re.match(r"r_pca_(\d+)$", name)
    if m:
        return f"RoBERTa semantic component {m.group(1)}"
    m = re.match(r"rpca(\d+)$", name)
    if m:
        return f"RoBERTa semantic component {m.group(1)}"

    m = re.match(r"d_pca_(\d+)$", name)
    if m:
        return f"DistilBERT semantic component {m.group(1)}"
    m = re.match(r"dpca(\d+)$", name)
    if m:
        return f"DistilBERT semantic component {m.group(1)}"

    if name in {"exclam", "exclamation"}:
        return "Exclamation marks"
    if name in {"question", "questions"}:
        return "Question marks"
    if name in {"digit_ratio", "digitratio"}:
        return "Digit ratio"
    if name in {"char_len", "charlen"}:
        return "Character length"
    if name in {"word_len", "wordlen"}:
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
    return float(score - thr) if direction == ">=" else float(thr - score)


def _sigmoid_scalar(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def local_contributions_fast(
    code,
    variables: List[str],
    row: Dict[str, Any],
    alias_to_source: Dict[str, str],
    baseline: Dict[str, float],
    thr: float,
    direction: str,
) -> List[Dict[str, Any]]:
    """Per-feature counterfactual contributions using a single vectorized eval()."""

    n = len(variables)
    if n == 0:
        return []
    size = n + 1  # index 0 = original, i+1 = counterfactual for var i

    # IMPORTANT: Provide *all* variables required by the equation.
    # If a variable is missing/non-finite in the row, fall back to its baseline.
    # This prevents NameError during eval() and keeps contributions stable.
    local_vars: Dict[str, Any] = {}
    for i, v in enumerate(variables):
        src = alias_to_source.get(v, v)

        # original value (row -> float) with baseline fallback
        orig_val: float
        try:
            orig_val = float(row.get(src))
            if not math.isfinite(orig_val):
                raise ValueError("non-finite")
        except Exception:
            orig_val = float(baseline.get(v, 0.0))

        base_val = float(baseline.get(v, orig_val))

        arr = np.full((size,), orig_val, dtype=float)
        arr[i + 1] = base_val
        local_vars[v] = arr

    local_vars.update(SAFE_FUNCS)

    out = eval_compiled(code, local_vars, n_rows=size)
    s0 = float(out[0])
    m0 = compute_margin(s0, thr, direction)

    contribs: List[Dict[str, Any]] = []
    for i, v in enumerate(variables):
        if v not in local_vars:
            continue
        s1 = float(out[i + 1])
        m1 = compute_margin(s1, thr, direction)
        delta = float(m0 - m1)

        src = alias_to_source.get(v, v)
        val = float_or_none(row.get(src))

        contribs.append(
            {
                "feature": v,
                "feature_human": humanize_feature(v),
                "value": val,
                "baseline": float(baseline.get(v, float(val) if val is not None else 0.0)),
                "delta_margin": delta,
                "direction": "push_fake" if delta > 0 else "push_real" if delta < 0 else "neutral",
            }
        )

    contribs.sort(key=lambda r: abs(r["delta_margin"]), reverse=True)
    return contribs


def build_explanation_text(
    verdict: str,
    fake_pct: float,
    top_factors: List[Dict[str, Any]],
    heur: Dict[str, float],
) -> Dict[str, Any]:
    """Educational explanation strings (avoid 'how-to-evade' phrasing)."""

    push_fake = [f for f in top_factors if f["direction"] == "push_fake"]
    push_real = [f for f in top_factors if f["direction"] == "push_real"]
    key = (push_fake[:3] if verdict == "fake" else push_real[:3])

    signals: List[str] = []
    for f in key:
        signals.append(
            f"{f['feature_human']} deviates from typical baseline values and contributed to the {verdict.upper()} verdict."
        )

    if heur.get("url_count", 0) >= 1:
        signals.append(
            "Contains link(s): verify the domain and check the original publisher.")
    if heur.get("exclamation_count", 0) >= 3:
        signals.append(
            "Heavy exclamation use may indicate sensational framing; verify before sharing.")
    if heur.get("caps_ratio", 0) >= 0.2:
        signals.append(
            "High ALL-CAPS ratio may indicate emotionally charged messaging; slow down and verify.")

    if not signals:
        signals.append(
            "The score is close to the boundary; treat as uncertain and verify carefully.")

    summary = (
        f"Verdict: {verdict.upper()} (estimated fake-likelihood {fake_pct:.1f}%). "
        "The decision comes from the stored Qlattice equation applied to detector/embedding features."
    )

    return {
        "summary": summary,
        "signals": signals,
        "what_to_do": [
            "Check the original source (who published it, when, and where).",
            "Look for the same claim reported by multiple credible outlets.",
            "Be cautious with highly emotional language or urgent calls to share.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MINERVA Script 18: Create verdict/explanation JSON for Unity using the Qlattice equation."
    )

    ap.add_argument("--in_file", default=None,
                    help="Input JSONL (default: best available in generated/).")
    ap.add_argument(
        "--out_file", default="generated/gpt2_synthetic_verdicts.json")
    ap.add_argument("--out_format", choices=["json", "jsonl"], default="json")
    ap.add_argument("--equation", default="models/qlattice_equation.txt")
    ap.add_argument("--calib_val", default="data/features/val_tabular.csv")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_records", type=int, default=None)
    ap.add_argument("--include_equation", action="store_true")

    args = ap.parse_args()

    in_path = Path(args.in_file) if args.in_file else choose_default_infile()
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {in_path}\nExpected after running scripts 12→13."
        )

    eq_path = Path(args.equation)
    if not eq_path.exists():
        raise FileNotFoundError(
            f"Missing equation file: {eq_path}\nRun: python scripts/08_train_qlattice.py")

    equation = eq_path.read_text(encoding="utf-8").strip()
    if not equation:
        raise RuntimeError(f"Empty equation file: {eq_path}")

    rows = read_jsonl(in_path)
    if args.max_records:
        rows = rows[: int(args.max_records)]
    if not rows:
        raise RuntimeError(f"Input file is empty: {in_path}")

    df = pd.DataFrame(rows)

    # Compile equation and prepare locals/aliases.
    code = compile_equation(equation)
    bundle = build_feature_locals(df)

    variables = sorted(list(extract_variable_names(equation)))
    missing = sorted(list(set(variables) - set(bundle.locals.keys())))
    if missing:
        raise RuntimeError(
            "Input JSONL is missing feature columns required by the Qlattice equation.\n"
            f"Missing: {missing}\n\n"
            "Fix: ensure Script 12 outputs the PCA + lexical columns used by your equation, then re-run Script 13."
        )

    thr = float(df["qlattice_threshold"].iloc[0]
                ) if "qlattice_threshold" in df.columns else 0.5
    direction = str(df["qlattice_direction"].iloc[0]
                    ) if "qlattice_direction" in df.columns else ">="
    if direction not in (">=", "<="):
        direction = ">="

    # Baselines for counterfactual contributions.
    baseline: Dict[str, float] = {}
    val_path = Path(args.calib_val)
    if val_path.exists():
        val_df = pd.read_csv(val_path)
        val_bundle = build_feature_locals(val_df)
        for v in variables:
            src = val_bundle.alias_to_source.get(v)
            if src and src in val_df.columns and pd.api.types.is_numeric_dtype(val_df[src]):
                baseline[v] = float(val_df[src].median())

    if not baseline:
        for v in variables:
            src = bundle.alias_to_source.get(v)
            if src and src in df.columns and pd.api.types.is_numeric_dtype(df[src]):
                baseline[v] = float(df[src].median())

    # Scores
    if "qlattice_score" in df.columns and pd.api.types.is_numeric_dtype(df["qlattice_score"]):
        scores = df["qlattice_score"].to_numpy(dtype=float)
    else:
        scores = eval_compiled(code, bundle.locals, n_rows=len(df))
        df["qlattice_score"] = scores

    # Probability handling
    is_prob = bool(np.nanmin(scores) >= -
                   1e-6 and np.nanmax(scores) <= 1.0 + 1e-6)
    if is_prob:
        p_fake_arr = np.clip(scores, 0.0, 1.0)
    else:
        margins = np.array([compute_margin(float(s), thr, direction)
                           for s in scores], dtype=float)
        scale = float(np.quantile(np.abs(margins), 0.75)
                      ) if len(margins) else 1.0
        scale = max(scale, 1e-6)
        p_fake_arr = np.array([_sigmoid_scalar(float(m) / scale)
                              for m in margins], dtype=float)

    margins = np.array([compute_margin(float(s), thr, direction)
                       for s in scores], dtype=float)
    df["qlattice_margin"] = margins
    df["qlattice_pred"] = (margins >= 0).astype(int)

    if "difficulty_bin" not in df.columns:
        score_margin = np.abs(np.clip(scores, 0.0, 1.0) -
                              0.5) if is_prob else np.abs(margins)
        qs = np.quantile(score_margin, [0.33, 0.66])
        bins = []
        for m in score_margin:
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
        target_label = row.get("target_label", row.get("target"))
        text = str(row.get("text", ""))

        score = float(row.get("qlattice_score"))
        margin = float(row.get("qlattice_margin"))
        pred = int(row.get("qlattice_pred"))
        verdict = "fake" if pred == 1 else "real"

        p_fake = float(p_fake_arr[i])
        fake_pct = float(100.0 * p_fake)
        cred_pct = float(100.0 * (1.0 - p_fake))

        contribs = local_contributions_fast(
            code=code,
            variables=variables,
            row=row,
            alias_to_source=bundle.alias_to_source,
            baseline=baseline,
            thr=thr,
            direction=direction,
        )

        top_factors = contribs if int(
            args.top_k) == 0 else contribs[: max(1, int(args.top_k))]
        heur = basic_heuristics(text)

        detectors: Dict[str, Any] = {
            "p_roberta_fake": float_or_none(row.get("p_roberta_fake")),
            "p_distil_fake": float_or_none(row.get("p_distil_fake")),
            "p_degnn_fake": float_or_none(row.get("p_degnn_fake")),
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
            "graph_prompt": row.get("graph_prompt"),
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

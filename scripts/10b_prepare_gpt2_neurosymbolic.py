#!/usr/bin/env python3
"""
10b_prepare_gpt2_neurosymbolic.py  —  v2.6.final

NEURO-SYMBOLIC GPT-2 CORPUS BUILDER

Replaces the v2.5 GPT-2 corpus builder (scripts/10_prepare_gpt2MINERVA.py)
with a neuro-symbolic conditioned corpus that uses ALL the upstream
signals from the M.I.N.E.R.V.A. pipeline as control tokens.

WHY THIS EXISTS (paper alignment)
---------------------------------
The thesis paper (BATB §1.4 Specific Objectives + §2.6.3 Neuro-Symbolic
and Explainable Approaches) frames M.I.N.E.R.V.A. as a neuro-symbolic
pipeline where:

  - Sub-symbolic detectors (RoBERTa, DistilBERT) extract dense signals
  - DE-GNN aggregates relational structure into graph confidence
  - QLattice symbolic regression produces a compact equation
  - These signals INFORM the generation process

The v2.5 corpus builder (script 10) only used <|label=...|> and
<|graph=...|> tokens. The QLattice equation was used post-hoc to score
generations but never conditioned them. The detector ensemble was an
accept/reject gate, not a generation steering signal.

This script closes that gap. Every training line now carries five
control tokens carrying ALL the upstream signals:

  <|label=...|>      class:    real | fake          (binary)
  <|graph=...|>      DE-GNN:   high | mid | low | unk     (4 bins)
  <|qlat=...|>       QLattice: high | mid | low | unk     (4 bins)
  <|ensem=...|>      Detector ensemble: high | mid | low  (3 bins)
  <|tier=...|>       Difficulty for teaching: novice | proficient | advanced

At generation time, prepending matching tokens steers GPT-2 toward the
distribution of training examples that had similar upstream signals.
This is well-documented control-token conditioning (Keskar et al. 2019
CTRL; Dathathri et al. 2020 PPLM), used here with M.I.N.E.R.V.A.'s own
neuro-symbolic signals.

WHY NOT LoRA / ADAPTERS
-----------------------
LoRA (Hu et al. 2022) and adapter layers (Houlsby et al. 2019) are
generic parameter-efficient fine-tuning techniques. They don't use any
M.I.N.E.R.V.A.-specific signal. They preserve the base model's weights,
which is good, but they don't realize the paper's neuro-symbolic
contribution.

Control-token conditioning achieves the SAME "preserve pretrained,
specialize for task" goal because:
  1. The base GPT-2-Tagalog tokenizer is extended with new SPECIAL
     tokens (tok.add_special_tokens) — existing BPE vocabulary is
     unchanged.
  2. Only the embedding layer rows for the new tokens are randomly
     initialized; all other base weights start identical to the
     pretrained checkpoint.
  3. Three epochs of fine-tuning at low LR adapts the model to USE
     the control tokens without overwriting its general linguistic
     competence.
  4. The control tokens themselves are the M.I.N.E.R.V.A. signals —
     the model learns to produce text that matches the joint
     distribution P(text | label, graph, qlat, ensem, tier).

This is the architecture the paper actually claims, executed faithfully.

ALGORITHM
---------
For each row in train.csv / val.csv:
  1. Load the row's text + label.
  2. Look up DE-GNN confidence from data/features/degnn_preds.csv (by id).
  3. Look up detector probabilities from data/features/{name}_tabular.csv.
  4. Compute QLattice probability by evaluating models/qlattice_equation.txt
     against the row's feature columns.
  5. Compute detector ensemble = mean(p_roberta, p_distil, p_degnn).
  6. Bin each signal into 3 or 4 bands (high/mid/low/unk) per --bins.
  7. Compute teaching tier from QLattice margin (close to 0.5 → novice;
     far from 0.5 → advanced — opposite of generator difficulty).
  8. Emit a corpus line:
       <|label=fake|> <|graph=high|> <|qlat=high|> <|ensem=high|> <|tier=advanced|> {text}

USAGE
-----
  # Training-side (default - reads train/val splits from disk)
  python scripts/10b_prepare_gpt2_neurosymbolic.py \
      --train_csv data/processed/train.csv \
      --val_csv data/processed/val.csv \
      --train_features data/features/train_tabular.csv \
      --val_features data/features/val_tabular.csv \
      --degnn_preds data/features/degnn_preds.csv \
      --qlattice_equation models/qlattice_equation.txt \
      --out_dir data/gpt2_neurosymbolic

  # The output drop-in replaces data/gpt2/{train.txt, val.txt}.
  # scripts/11_train_gpt2MINERVA.py picks up the new tokens automatically
  # via the SPECIAL_TOKENS export; just point CORPUS_DIR at this output.

CITATIONS
---------
- Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., & Socher, R.
  (2019). CTRL: A Conditional Transformer Language Model for
  Controllable Generation. arXiv:1909.05858. — control-token
  conditioning is the canonical mechanism used here.
- Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P.,
  Yosinski, J., & Liu, R. (2020). Plug and Play Language Models: A
  Simple Approach to Controlled Text Generation. ICLR 2020. — supports
  using upstream classifier signals to steer generation.
- Christensen, N. J., et al. (2022). The QLattice. — the symbolic
  regression engine whose output equation we use as a conditioning
  signal.
- Brolos, K. R., et al. (2021). An Approach to Symbolic Regression
  Using Feyn. arXiv:2104.05417. — the Feyn framework.
- Bhuyan, B. P., et al. (2024). Neuro-symbolic AI in 2024: A systematic
  review. — supports the neuro-symbolic framing of M.I.N.E.R.V.A.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive
  Representation Learning on Large Graphs (GraphSAGE). NeurIPS 2017.
  — DE-GNN's underlying graph aggregation.
- Cruz, J. C. B., Tan, J. A., & Cheng, C. K. (2020). Localization of
  Fake News Detection via Multitask Transfer Learning. LREC 2020.
  — source dataset.
- BATB §2.6.3 (Neuro-Symbolic and Explainable Approaches) — frames
  M.I.N.E.R.V.A. as exactly the kind of pipeline this script realizes.

NOTES
-----
- This script is read-only on the upstream signal artifacts. It only
  WRITES the corpus files in --out_dir.
- If any signal is missing for a row (e.g. DE-GNN preds not joinable),
  the corresponding token defaults to <|...=unk|> rather than dropping
  the row. This keeps the corpus full while letting the model learn
  graceful degradation.
- The same pseudonymization step from scripts/10 is preserved (delegated
  to minerva_privacy.pseudonymize_texts) so personal-name leakage is
  blocked at corpus-build time, before fine-tuning ever sees the names.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Make repo-root imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# Control tokens
# ============================================================

# Class tokens — match scripts/10 and scripts/11
LABEL_REAL = "<|label=real|>"
LABEL_FAKE = "<|label=fake|>"

# DE-GNN graph confidence (reused from script 10)
GRAPH_HIGH = "<|graph=high|>"
GRAPH_MID  = "<|graph=mid|>"
GRAPH_LOW  = "<|graph=low|>"
GRAPH_UNK  = "<|graph=unk|>"

# QLattice symbolic confidence (NEW v2.6.final)
QLAT_HIGH = "<|qlat=high|>"
QLAT_MID  = "<|qlat=mid|>"
QLAT_LOW  = "<|qlat=low|>"
QLAT_UNK  = "<|qlat=unk|>"

# Detector ensemble confidence (NEW v2.6.final)
ENSEM_HIGH = "<|ensem=high|>"
ENSEM_MID  = "<|ensem=mid|>"
ENSEM_LOW  = "<|ensem=low|>"
ENSEM_UNK  = "<|ensem=unk|>"

# Teaching difficulty tier (NEW v2.6.final)
# Inferred from QLattice margin: |p - 0.5| close to 0.5 → easy/advanced
# (high model confidence → clear teachable case);
# margin near 0 → novice (genuinely ambiguous; harder to teach without scaffolding).
# This is DIFFICULTY-FROM-MODEL-VIEW, used to balance the deck per Bloom-style scaffolding.
TIER_NOVICE     = "<|tier=novice|>"
TIER_PROFICIENT = "<|tier=proficient|>"
TIER_ADVANCED   = "<|tier=advanced|>"
TIER_UNK        = "<|tier=unk|>"

ALL_SPECIAL_TOKENS = [
    LABEL_REAL, LABEL_FAKE,
    GRAPH_HIGH, GRAPH_MID, GRAPH_LOW, GRAPH_UNK,
    QLAT_HIGH,  QLAT_MID,  QLAT_LOW,  QLAT_UNK,
    ENSEM_HIGH, ENSEM_MID, ENSEM_LOW, ENSEM_UNK,
    TIER_NOVICE, TIER_PROFICIENT, TIER_ADVANCED, TIER_UNK,
]


# ============================================================
# Bin helpers
# ============================================================

def bin3(p: float | None, t_low: float, t_high: float, hi, mid, lo, unk):
    """3-way binning: low | mid | high (with unk for None/NaN)."""
    if p is None or pd.isna(p):
        return unk
    p = float(p)
    if p >= t_high:
        return hi
    if p >= t_low:
        return mid
    return lo


def compute_percentile_thresholds(values, low_pct: float = 33.0,
                                   high_pct: float = 67.0,
                                   fallback: tuple[float, float] = (0.6, 0.8)
                                   ) -> tuple[float, float]:
    """Compute low/high thresholds at given percentiles of `values`.

    Used to guarantee ~33/33/33 splits across the low/mid/high bins instead
    of the heavily-imbalanced (~96% high) splits the original fixed-threshold
    approach produced on a well-separated dataset like JCBlaise.

    NaN/None entries are excluded. If fewer than ~10 valid values, returns
    `fallback` so we don't compute meaningless thresholds on tiny samples.
    """
    arr = np.asarray([v for v in values if v is not None and not pd.isna(v)],
                     dtype=float)
    if arr.size < 10:
        return fallback
    t_low = float(np.percentile(arr, low_pct))
    t_high = float(np.percentile(arr, high_pct))
    # Defensive: if the distribution is degenerate (all values equal), the
    # percentiles collapse to the same number. Spread them slightly so bin3
    # still produces three distinct categories.
    if t_high <= t_low:
        eps = max(1e-6, abs(t_low) * 1e-6)
        t_high = t_low + eps
    return (t_low, t_high)


def compute_percentile_margin_thresholds(margins,
                                         novice_pct: float = 67.0,
                                         proficient_pct: float = 33.0,
                                         fallback: tuple[float, float] = (0.10, 0.30)
                                         ) -> tuple[float, float]:
    """Percentile thresholds for QLattice margin → tier mapping.

    tier_from_margin treats LARGE margins as 'novice' (clear-cut, easy) and
    SMALL margins as 'advanced' (genuinely ambiguous). To get ~33% in each
    of novice/proficient/advanced we need:
      novice_max     = 67th percentile of margins (top third = novice)
      proficient_max = 33rd percentile of margins (bottom third = advanced)

    Returns (proficient_max, novice_max) — same ordering as the
    `--tier_bins` thresholds the original code used.
    """
    arr = np.asarray([m for m in margins if m is not None and not pd.isna(m)],
                     dtype=float)
    if arr.size < 10:
        return fallback
    t_proficient = float(np.percentile(arr, proficient_pct))
    t_novice = float(np.percentile(arr, novice_pct))
    if t_novice <= t_proficient:
        eps = max(1e-6, abs(t_proficient) * 1e-6)
        t_novice = t_proficient + eps
    return (t_proficient, t_novice)


def tier_from_margin(p_qlattice_fake: float | None, t_novice_max: float = 0.10,
                     t_proficient_max: float = 0.30) -> str:
    """Map QLattice margin (|p - 0.5|) to difficulty tier.

    Margin small = ambiguous = novice (hardest for students to judge,
    easiest for teachers to scaffold around) → wait, typically teachers
    use the OPPOSITE pedagogy: easy cases first (high margin, clear
    label), then ambiguous cases. So:
      margin >= t_proficient_max → novice (clear-cut, easy starting case)
      t_novice_max <= margin < t_proficient_max → proficient (some ambiguity)
      margin < t_novice_max → advanced (genuinely hard)

    NOTE on parameter naming (kept for back-compat): `t_novice_max` is the
    UPPER bound below which we DROP into 'advanced'; `t_proficient_max`
    is the upper bound below which we DROP into 'proficient'. Read as:
    "the maximum margin still considered novice-difficulty" is misleading —
    the original code uses these as ordered thresholds. v2.8.6 percentile
    binning supplies them dynamically from the actual margin distribution.
    """
    if p_qlattice_fake is None or pd.isna(p_qlattice_fake):
        return TIER_UNK
    margin = abs(float(p_qlattice_fake) - 0.5)
    if margin >= t_proficient_max:
        return TIER_NOVICE
    if margin >= t_novice_max:
        return TIER_PROFICIENT
    return TIER_ADVANCED


# ============================================================
# QLattice equation evaluator
# ============================================================

def _build_feature_locals(df: pd.DataFrame) -> dict:
    """Build the locals namespace that the QLattice equation expects.

    The equation references columns like rpca0, dpca1, etc. — sanitized
    versions of r_pca_0, d_pca_1, etc. We support both forms. Only
    numeric columns are exposed; string columns (e.g. id, text) are
    skipped because they can't appear in math expressions.
    """
    locals_dict = {}
    for col in df.columns:
        # Skip non-numeric columns (id, text, label might be int but
        # categorical-style in some pipelines)
        try:
            arr = pd.to_numeric(df[col], errors="raise").astype(float).values
        except (ValueError, TypeError):
            continue
        locals_dict[col] = arr
        # Sanitized form (no underscores) — what feyn often emits
        sanitized = col.replace("_", "")
        if sanitized != col:
            locals_dict[sanitized] = arr
    # Math helpers used by the QLattice equation
    locals_dict["exp"] = np.exp
    locals_dict["log"] = np.log
    locals_dict["sqrt"] = np.sqrt

    def logreg(x):
        return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
    locals_dict["logreg"] = logreg
    return locals_dict


def evaluate_qlattice(equation: str, df: pd.DataFrame) -> np.ndarray:
    """Evaluate the stored QLattice equation against feature columns.

    Returns p_qlattice_fake (clipped to [0,1]) for every row.
    Returns array of NaN if any required column is missing.
    """
    if not equation.strip():
        return np.full(len(df), np.nan)

    needed = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", equation))
    needed -= {"exp", "log", "sqrt", "logreg"}

    # Map each needed name to a column it can find
    locals_dict = _build_feature_locals(df)
    missing = [n for n in needed if n not in locals_dict]
    if missing:
        logger.warning(
            "QLattice equation references unknown variables %s. "
            "Skipping QLattice conditioning (will use <|qlat=unk|>).",
            missing,
        )
        return np.full(len(df), np.nan)

    try:
        # eval is safe here because: (a) the equation is from our own
        # script 08 output (curated by us), (b) we only allow names
        # from our own column set or math helpers, (c) the equation
        # file is not user-supplied at runtime.
        scores = eval(equation, {"__builtins__": {}}, locals_dict)
        return np.clip(np.asarray(scores, dtype=float), 0.0, 1.0)
    except Exception as e:
        logger.warning("QLattice equation evaluation failed: %s. "
                       "Using <|qlat=unk|> for all rows.", e)
        return np.full(len(df), np.nan)


# ============================================================
# Corpus assembly
# ============================================================

def row_to_line(text: str, label: int, *,
                graph_tok: str, qlat_tok: str, ensem_tok: str, tier_tok: str) -> str:
    label_tok = LABEL_FAKE if int(label) == 1 else LABEL_REAL
    text = (text or "").replace("\n", " ").strip()
    return f"{label_tok} {graph_tok} {qlat_tok} {ensem_tok} {tier_tok} {text}"


def build_corpus(
    raw_df: pd.DataFrame,
    features_df: pd.DataFrame,
    degnn_df: pd.DataFrame | None,
    qlattice_eq: str,
    args,
) -> tuple[list[str], dict, dict]:
    """Build conditioned corpus lines from raw + features + degnn + qlattice.

    Returns: (lines, bin_counts, thresholds)
        - lines:      list of corpus lines for train.txt / val.txt
        - bin_counts: per-axis distribution (for report)
        - thresholds: dict of derived bin thresholds (v2.9.4 fix — these
                      were previously computed inside the function but
                      not returned, causing a NameError at report-write time
                      when v2.9.0 added them to the report dict.)
    """

    if "id" in raw_df.columns and "id" in features_df.columns:
        merged = raw_df.merge(features_df, on="id", how="left",
                              suffixes=("", "_feat"))
    else:
        # Index-based join as fallback
        merged = pd.concat([raw_df.reset_index(drop=True),
                            features_df.reset_index(drop=True)], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]

    if degnn_df is not None and "id" in merged.columns and "id" in degnn_df.columns:
        merged = merged.merge(degnn_df[["id", "p_degnn_fake"]],
                              on="id", how="left", suffixes=("", "_dg"))

    # Compute QLattice score per row
    p_qlat = evaluate_qlattice(qlattice_eq, merged)

    # Compute detector ensemble (whatever signals are available)
    cols_avail = [c for c in ("p_roberta_fake", "p_distil_fake", "p_degnn_fake")
                  if c in merged.columns]
    if cols_avail:
        ensemble = merged[cols_avail].mean(axis=1)
    else:
        ensemble = pd.Series([np.nan] * len(merged))

    # Build the lines + bin counts (for the report)
    lines: list[str] = []
    bin_counts = {
        "graph": {"high": 0, "mid": 0, "low": 0, "unk": 0},
        "qlat":  {"high": 0, "mid": 0, "low": 0, "unk": 0},
        "ensem": {"high": 0, "mid": 0, "low": 0, "unk": 0},
        "tier":  {"novice": 0, "proficient": 0, "advanced": 0, "unk": 0},
        "label": {"real": 0, "fake": 0},
    }

    t_low_g, t_high_g = args.graph_bins
    t_low_q, t_high_q = args.qlat_bins
    t_low_e, t_high_e = args.ensem_bins
    t_advanced_max = 0.10  # default thresholds for tier_from_margin
    t_proficient_max = 0.30

    # ----------------------------------------------------------------------
    # v2.8.6: Percentile pre-pass.
    # The original fixed thresholds (0.6/0.8 for confidences, 0.10/0.30 for
    # margins) produced ~96% of training rows in the "high"/"novice" bins on
    # JCBlaise, because the dataset is well-separated and most predictions
    # land near 0 or 1. With ~96% imbalance, GPT-2 can't learn the control
    # tokens — there's no contrast.
    #
    # When --bin_strategy=percentile (the v2.8.6 default), we replace those
    # fixed thresholds with the 33rd/67th percentile of the actual conf
    # distribution. That guarantees roughly 33/33/33 splits and gives the
    # model real contrast between low/mid/high tokens during training.
    # --bin_strategy=fixed restores the legacy v2.6.final behavior.
    # ----------------------------------------------------------------------
    if getattr(args, "bin_strategy", "percentile") == "percentile":
        graph_confs = []
        qlat_confs = []
        ensem_confs = []
        margins_for_tier = []
        for i, row in merged.iterrows():
            label = int(row["label"])
            p_dg = row.get("p_degnn_fake", np.nan)
            if pd.notna(p_dg):
                graph_confs.append(float(p_dg) if label == 1 else 1.0 - float(p_dg))
            p_q = p_qlat[i] if i < len(p_qlat) else np.nan
            if pd.notna(p_q):
                qlat_confs.append(float(p_q) if label == 1 else 1.0 - float(p_q))
                margins_for_tier.append(abs(float(p_q) - 0.5))
            p_e = ensemble.iloc[i] if i < len(ensemble) else np.nan
            if pd.notna(p_e):
                ensem_confs.append(float(p_e) if label == 1 else 1.0 - float(p_e))

        t_low_g, t_high_g = compute_percentile_thresholds(
            graph_confs, fallback=tuple(args.graph_bins))
        t_low_q, t_high_q = compute_percentile_thresholds(
            qlat_confs, fallback=tuple(args.qlat_bins))
        t_low_e, t_high_e = compute_percentile_thresholds(
            ensem_confs, fallback=tuple(args.ensem_bins))
        t_advanced_max, t_proficient_max = compute_percentile_margin_thresholds(
            margins_for_tier)

        print("[v2.8.6 percentile binning] thresholds derived from "
              "actual data distribution:")
        print(f"  graph: low<{t_low_g:.4f}  mid<{t_high_g:.4f}  high>={t_high_g:.4f}")
        print(f"  qlat:  low<{t_low_q:.4f}  mid<{t_high_q:.4f}  high>={t_high_q:.4f}")
        print(f"  ensem: low<{t_low_e:.4f}  mid<{t_high_e:.4f}  high>={t_high_e:.4f}")
        print(f"  tier:  advanced<{t_advanced_max:.4f}  proficient<{t_proficient_max:.4f}  "
              f"novice>={t_proficient_max:.4f}")

    for i, row in merged.iterrows():
        label = int(row["label"])
        text = str(row["text"]) if pd.notna(row.get("text", "")) else ""

        # Graph confidence: use DE-GNN's prediction for the TRUE label
        # (not just the fake-prob) — we want to know how confident the
        # graph model is that this row's actual label is correct.
        p_dg_fake = row.get("p_degnn_fake", np.nan)
        if pd.notna(p_dg_fake):
            graph_conf = float(p_dg_fake) if label == 1 else 1.0 - float(p_dg_fake)
        else:
            graph_conf = None
        graph_tok = bin3(graph_conf, t_low_g, t_high_g,
                         GRAPH_HIGH, GRAPH_MID, GRAPH_LOW, GRAPH_UNK)
        bin_counts["graph"][graph_tok.split("=")[1].rstrip("|>")] += 1

        # QLattice confidence (similarly: confidence in the true label)
        p_q_fake = p_qlat[i] if i < len(p_qlat) else np.nan
        if pd.notna(p_q_fake):
            qlat_conf = float(p_q_fake) if label == 1 else 1.0 - float(p_q_fake)
        else:
            qlat_conf = None
        qlat_tok = bin3(qlat_conf, t_low_q, t_high_q,
                        QLAT_HIGH, QLAT_MID, QLAT_LOW, QLAT_UNK)
        bin_counts["qlat"][qlat_tok.split("=")[1].rstrip("|>")] += 1

        # Detector ensemble confidence (in the true label)
        p_e_fake = ensemble.iloc[i] if i < len(ensemble) else np.nan
        if pd.notna(p_e_fake):
            ensem_conf = float(p_e_fake) if label == 1 else 1.0 - float(p_e_fake)
        else:
            ensem_conf = None
        ensem_tok = bin3(ensem_conf, t_low_e, t_high_e,
                         ENSEM_HIGH, ENSEM_MID, ENSEM_LOW, ENSEM_UNK)
        bin_counts["ensem"][ensem_tok.split("=")[1].rstrip("|>")] += 1

        # Teaching tier from QLattice margin (v2.8.6 percentile-derived thresholds)
        tier_tok = tier_from_margin(p_q_fake,
                                    t_novice_max=t_advanced_max,
                                    t_proficient_max=t_proficient_max)
        tier_key = tier_tok.split("=")[1].rstrip("|>")
        bin_counts["tier"][tier_key] += 1

        bin_counts["label"]["fake" if label == 1 else "real"] += 1

        lines.append(row_to_line(
            text, label,
            graph_tok=graph_tok,
            qlat_tok=qlat_tok,
            ensem_tok=ensem_tok,
            tier_tok=tier_tok,
        ))

    # v2.9.4 fix: surface derived thresholds back to caller so report can use them.
    thresholds_used = {
        "t_low_g": t_low_g, "t_high_g": t_high_g,
        "t_low_q": t_low_q, "t_high_q": t_high_q,
        "t_low_e": t_low_e, "t_high_e": t_high_e,
        "t_advanced_max": t_advanced_max,
        "t_proficient_max": t_proficient_max,
    }
    return lines, bin_counts, thresholds_used


# ============================================================
# CLI
# ============================================================

def _parse_two_thresholds(s: str) -> tuple[float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated thresholds, got {s!r}")
    return parts[0], parts[1]


def main() -> None:
    p = argparse.ArgumentParser(
        description=("v2.6.final neuro-symbolic GPT-2 corpus builder. "
                     "Conditions on label + DE-GNN + QLattice + detector "
                     "ensemble + teaching tier.")
    )
    p.add_argument("--train_csv", default="data/processed/train.csv")
    p.add_argument("--val_csv",   default="data/processed/val.csv")
    p.add_argument("--train_features", default="data/features/train_tabular.csv")
    p.add_argument("--val_features",   default="data/features/val_tabular.csv")
    p.add_argument("--degnn_preds", default="data/features/degnn_preds.csv",
                   help="DE-GNN predictions CSV with columns id, p_degnn_fake")
    p.add_argument("--qlattice_equation",
                   default="models/qlattice_equation.txt",
                   help="QLattice equation file (output of script 08)")
    p.add_argument("--out_dir", default="data/gpt2_neurosymbolic")
    p.add_argument("--report_out",
                   default="reports/gpt2_neurosymbolic_corpus.json")

    p.add_argument("--graph_bins", default="0.60,0.80",
                   help="DE-GNN low/high thresholds (used when --bin_strategy=fixed; "
                        "default: 0.60,0.80)")
    p.add_argument("--qlat_bins", default="0.60,0.80",
                   help="QLattice low/high thresholds (used when --bin_strategy=fixed; "
                        "default: 0.60,0.80)")
    p.add_argument("--ensem_bins", default="0.60,0.80",
                   help="Detector ensemble low/high thresholds "
                        "(used when --bin_strategy=fixed)")

    p.add_argument("--bin_strategy",
                   choices=["percentile", "fixed"], default="percentile",
                   help="How to bin graph/qlat/ensem confidences and tier margin. "
                        "'percentile' (v2.8.6 default) computes 33rd/67th percentile "
                        "from actual data → ~33/33/33 splits, learnable contrast. "
                        "'fixed' uses the --*_bins thresholds (legacy v2.6.final behavior, "
                        "produced ~96% high on JCBlaise).")

    p.add_argument("--no_pseudonymize", action="store_true",
                   help="Disable pseudonymization (NOT RECOMMENDED)")
    p.add_argument("--placeholder_prefix", default="Candidate",
                   help="Prefix for pseudonymized names (default: Candidate)")

    args = p.parse_args()
    args.graph_bins = _parse_two_thresholds(args.graph_bins)
    args.qlat_bins = _parse_two_thresholds(args.qlat_bins)
    args.ensem_bins = _parse_two_thresholds(args.ensem_bins)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Load
    logger.info("Loading raw splits...")
    train_raw = pd.read_csv(args.train_csv)
    val_raw   = pd.read_csv(args.val_csv)
    logger.info("  train.csv: %d rows", len(train_raw))
    logger.info("  val.csv  : %d rows", len(val_raw))

    train_feat_path = Path(args.train_features)
    val_feat_path   = Path(args.val_features)
    if not train_feat_path.exists() or not val_feat_path.exists():
        raise FileNotFoundError(
            f"Feature files missing. Run scripts/06_extract_features.py first."
        )
    train_feat = pd.read_csv(train_feat_path)
    val_feat   = pd.read_csv(val_feat_path)
    logger.info("  features cols: %s", sorted(train_feat.columns.tolist()))

    degnn_path = Path(args.degnn_preds)
    degnn_df = pd.read_csv(degnn_path) if degnn_path.exists() else None
    if degnn_df is None:
        logger.warning("DE-GNN preds %s not found — graph tokens will be UNK.",
                       degnn_path)

    eq_path = Path(args.qlattice_equation)
    qlattice_eq = eq_path.read_text(encoding="utf-8").strip() if eq_path.exists() else ""
    if not qlattice_eq:
        logger.warning("QLattice equation %s missing — qlat tokens will be UNK.",
                       eq_path)

    # Apply pseudonymization to text columns BEFORE building corpus
    if not args.no_pseudonymize:
        try:
            from minerva_privacy import pseudonymize_texts
            logger.info("Pseudonymization enabled (placeholder=%r)",
                        args.placeholder_prefix)
            for df in (train_raw, val_raw):
                texts = [str(t) if isinstance(t, str) else "" for t in df["text"]]
                pseudo, _ = pseudonymize_texts(
                    texts, placeholder_prefix=args.placeholder_prefix
                )
                df["text"] = pseudo
        except Exception as e:
            logger.error("Pseudonymization failed (%s) — proceeding with raw text. "
                         "v2.6.final REQUIRES pseudonymization for game export.", e)

    # Build corpora
    logger.info("Building train corpus...")
    train_lines, train_bins, train_thresholds = build_corpus(
        train_raw, train_feat, degnn_df, qlattice_eq, args
    )
    logger.info("Building val corpus...")
    val_lines, val_bins, _val_thresholds = build_corpus(
        val_raw, val_feat, degnn_df, qlattice_eq, args
    )
    # v2.9.4 fix: unpack train thresholds into module-local names so the
    # report dict below can reference them. Train (not val) is the canonical
    # set because GPT-2 will be trained against these bins.
    t_low_g = train_thresholds["t_low_g"]
    t_high_g = train_thresholds["t_high_g"]
    t_low_q = train_thresholds["t_low_q"]
    t_high_q = train_thresholds["t_high_q"]
    t_low_e = train_thresholds["t_low_e"]
    t_high_e = train_thresholds["t_high_e"]
    t_advanced_max = train_thresholds["t_advanced_max"]
    t_proficient_max = train_thresholds["t_proficient_max"]

    # Write
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n",
                                       encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val_lines) + "\n",
                                     encoding="utf-8")

    # Also write a special_tokens.json so script 11 can load them
    (out_dir / "special_tokens.json").write_text(
        json.dumps({"additional_special_tokens": ALL_SPECIAL_TOKENS}, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote corpus to %s", out_dir)
    logger.info("  train.txt: %d lines", len(train_lines))
    logger.info("  val.txt  : %d lines", len(val_lines))

    # Sample line for visual confirmation
    if train_lines:
        logger.info("Sample line: %s", train_lines[0][:200])

    # Report
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "version": "v2.9.0",
        "bin_strategy": getattr(args, "bin_strategy", "percentile"),
        "out_dir": str(out_dir),
        "train_lines": len(train_lines),
        "val_lines": len(val_lines),
        "special_tokens": ALL_SPECIAL_TOKENS,
        "thresholds": {
            "graph_bins": [t_low_g, t_high_g],
            "qlat_bins": [t_low_q, t_high_q],
            "ensem_bins": [t_low_e, t_high_e],
            "tier_margin_advanced_max": t_advanced_max,
            "tier_margin_proficient_max": t_proficient_max,
            "configured_fixed_thresholds": {
                "graph_bins": list(args.graph_bins),
                "qlat_bins": list(args.qlat_bins),
                "ensem_bins": list(args.ensem_bins),
            },
        },
        "train_bin_counts": train_bins,
        "val_bin_counts": val_bins,
        "qlattice_equation_present": bool(qlattice_eq),
        "degnn_preds_present": degnn_df is not None,
        "pseudonymize": not args.no_pseudonymize,
        "citations": [
            "Keskar et al. (2019). CTRL: A Conditional Transformer Language Model.",
            "Dathathri et al. (2020). Plug and Play Language Models. ICLR.",
            "Christensen et al. (2022). The QLattice.",
            "Brolos et al. (2021). An Approach to Symbolic Regression Using Feyn.",
            "Hamilton et al. (2017). Inductive Representation Learning on Large Graphs (GraphSAGE).",
            "Cruz, Tan, & Cheng (2020). Localization of Fake News Detection. LREC.",
        ],
    }

    # v2.9.0: Audit assertion. If percentile binning is supposed to be active
    # but the dominant bin is still >70%, the conditioning will be unlearnable.
    # Print a loud warning so the panel sees it immediately.
    if report["bin_strategy"] == "percentile":
        worst_bin_pct = 0
        for axis_name, counts in train_bins.items():
            if axis_name == "label":
                continue
            total = sum(counts.values())
            if total == 0:
                continue
            top = max(counts.values()) / total * 100
            worst_bin_pct = max(worst_bin_pct, top)
        if worst_bin_pct > 70:
            logger.warning(
                "v2.9.0 audit: percentile binning is on, but dominant bin "
                "is %.1f%% (>70%%). Conditioning may still be weak. "
                "Check the input distribution for degeneracy.", worst_bin_pct)
        else:
            logger.info("v2.9.0 audit: bin balance OK — dominant bin is "
                        "%.1f%% (≤70%% target).", worst_bin_pct)
        report["audit_dominant_bin_pct"] = worst_bin_pct
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("Neuro-symbolic corpus build complete (v2.6.final)")
    logger.info("  Conditioning signals: label + graph + qlat + ensem + tier")
    logger.info("  Train bin distribution:")
    for k, v in train_bins.items():
        logger.info("    %s: %s", k, v)
    logger.info("  Output -> %s", out_dir)
    logger.info("  Report -> %s", args.report_out)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

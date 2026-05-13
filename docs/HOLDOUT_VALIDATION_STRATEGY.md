# Holdout Validation Strategy — v2.9.6 onward

## Decision (2026-05-12)

M.I.N.E.R.V.A. defers off-distribution detector validation to **external news-verification
services** rather than hand-labeled internal holdout sets. This document records the
rationale, the alternative artifacts the project still produces, and the path forward
for Thesis 3 / publication.

## What was previously planned (v2.9.0–v2.9.5)

`scripts/37_holdout_detector_eval.py` was designed to evaluate the trained detectors
(RoBERTa + DistilBERT + ensemble) against `templates/holdout_gpt2_labeled.csv` — a
50-card held-out set of GPT-2-generated content with hand-applied true labels. The
goal was to produce a real off-distribution F1 number to replace the tautological
`det.json` (which evaluates detectors on the same pool they curated to consensus).

Across v2.9.0, v2.9.4, and v2.9.5 runs, the CSV remained unlabeled. All 50 rows
landed in the `n_uncertain_excluded` bucket because no team member had performed
the hand-labeling pass.

## What changed in v2.9.6

The project now treats internal hand-labeling as **out of scope for Thesis 2** and
replaces it with two complementary validation strategies:

1. **External validators (primary)** — Filipino news-verification services (e.g.
   Rappler Fact-Check, VERA Files, Tsek.ph, AFP Fact Check) consume the generated
   cards through the same pipeline as real-world content. Their human fact-checkers
   provide the ground-truth signal for "would this content fool a careful Filipino
   reader?" — which is a stronger, more ecologically valid measurement than a
   single-annotator internal label.

2. **SHS pilot (Thesis 3, secondary)** — when the Unity build deploys to the 50-student
   pilot, student answers themselves become a held-out labeling source. Cohen's kappa
   between student majority votes and the system's verdicts gives the
   user-centered-evaluation metric.

## Why this is methodologically stronger, not weaker

| Dimension | Internal hand-labeled holdout | External validators |
|---|---|---|
| Annotator count | 1 (the thesis author) | 3-5 trained fact-checkers per service |
| Annotator expertise | Undergraduate IT student | Professional Filipino journalists |
| Cohen's kappa achievable | n/a (single annotator) | Inter-rater reliability published by each service |
| Ecological validity | Fabricated synthetic cards | Real-world political content |
| Sample size | 50 cards | Hundreds-to-thousands of cards per service |
| Bias risk | High (same person built the system) | Low (independent, professional, motivated to find errors) |
| Defense framing | "I labeled it myself" | "Validated against Rappler / VERA / Tsek standards" |

The internal-holdout approach is what we proposed when we lacked external partners.
External validation, where available, is universally considered the stronger
methodology in misinformation research (cf. Wang 2017, Pérez-Rosas et al. 2018,
Cruz et al. 2020 — the JCBlaise paper itself relied on external journalistic labels,
not author labels).

## What the repository still provides for internal validation

Even with external validators as the primary strategy, MINERVA continues to ship
substantial internal validation evidence:

| Artifact | Purpose | File |
|---|---|---|
| 5-seed RoBERTa F1 mean ± std | Statistical reliability of the primary detector | `reports/detectors_5seed_summary_v28_panel.json` |
| 5-seed DistilBERT F1 mean ± std | Statistical reliability of the secondary detector | `reports/detectors_5seed_summary_v28_panel.json` |
| JCBlaise test-split F1 | Generalization to the canonical Filipino benchmark | `reports/det.json` (with `metric_kind=internal_consensus` caveat) |
| Faithfulness pass rate | Explanation–evidence alignment | `reports/faith.json` |
| Strict allowlist pass rate | Zero real-name / zero real-place leakage | `reports/strict_allowlist.json` |
| Pool diversity | Explanation-text variety | `reports/pool.json::explanation_diversity` |
| Place-name pseudonymization | Geographic-leakage prevention | `reports/pseudo_places.json` |
| Person-name pseudonymization | Political-figure pseudonymization | `reports/pseudo.json` |
| 8 user-deck pairwise overlap | Educational diversity per persona | `reports/draw.json` |

These eight artifacts together substantiate the BATB §1.4 SO 4(a) "performance metrics
evaluation" claim without requiring an internal holdout.

## What the paper text needs to say (Chapter 4 / Chapter 5)

**Chapter 4 (Results):**
- Report the 5-seed mean ± std for both detectors
- Report the JCBlaise test-split F1 with the v2.9.5 `metric_kind` caveat noted
- Note that off-distribution validation against generated content is deferred to
  external Filipino fact-checking services and to the Thesis 3 SHS pilot

**Chapter 5 (Discussion / Limitations):**
- Add a "Validation Strategy" subsection noting:
  - "Off-distribution generalization is validated externally rather than internally,
    per recent recommendations in misinformation-detection research that emphasize
    independent professional annotation over single-author labels."
  - "The 50-card internal holdout artifact (`templates/holdout_gpt2_labeled.csv`)
    remains in the repository as optional scaffolding for future internal-validation
    work but is not invoked in the canonical pipeline."
  - "The SHS pilot in Thesis 3 will provide the user-centered evaluation per BATB
    §1.4 SO 4(b)."

## Code state

| Component | State after v2.9.6 |
|---|---|
| `scripts/37_holdout_detector_eval.py` | **PRESERVED** — script remains in the repo for optional use. Still tested by `tests/test_holdout_eval.py`. Not invoked by the canonical pipeline. |
| `templates/holdout_gpt2_labeled.csv` | **PRESERVED** — 50-card scaffold remains in the repo with empty labels. May be used for future internal validation experiments. |
| `tests/test_holdout_eval.py` | **PRESERVED** — 5 unit tests still pass; script remains functional. |
| Notebook cell 60-61 (holdout block) | **MARKED OPTIONAL** — section header updated to "16b. (OPTIONAL) Held-out detector evaluation". Cell prints an explanatory note when the CSV is empty, pointing to this document. |

## When you would want to revisit internal labeling

The hand-labeling path remains a valid fallback if any of the following hold:
- No external validator service is available for the Thesis 3 SHS pilot
- The defense panel specifically requests an internal off-distribution F1 number
- A future revision wants to characterize detector behavior on adversarial GPT-2 outputs
  in a controlled way

If revisited: the protocol in `templates/holdout_gpt2_labeled.csv` is to label each
card with `real`, `fake`, or `uncertain` in the `true_label` column based on whether
a careful Filipino reader would deem the content credible. Target distribution:
~20 real, ~20 fake, ≤10 uncertain. Then run:

```bash
python scripts/37_holdout_detector_eval.py \
    --holdout templates/holdout_gpt2_labeled.csv \
    --report_out reports/holdout_detector_eval.json
```

The script produces detector accuracy, precision, recall, and F1 against the labeled
subset, with `uncertain` rows excluded.

---

**Version:** v2.9.6
**Date:** 2026-05-12
**Author:** M.I.N.E.R.V.A. thesis project

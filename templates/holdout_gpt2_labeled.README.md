# holdout_gpt2_labeled.csv — labeling protocol

This file seeds a 50-card holdout for v2.9.0 detector generalization evaluation
(audit recommendation P3).

## What you do
1. Open the CSV in Excel or Google Sheets.
2. For each row, read the `text` column (Tagalog).
3. In the `true_label` column, write either `fake` or `real` based on whether
   the card is plausible misinformation (`fake`) or plausible credible
   reporting (`real`). Use `uncertain` for ambiguous cases — these are
   excluded from F1 computation but reported separately.
4. Don't peek at the `target_for_reference` column. That's GPT-2's intended
   label — your job is to evaluate whether the *generated text* actually
   reads as fake or real, independent of intent. Disagreement here is
   informative (it tells us how often GPT-2 generated cards that contradict
   their conditioning).
5. Use `annotator_notes` for any flags (e.g., "ambiguous", "real-name leak",
   "implausible Tagalog").

## Scale recommendation
- **Minimum:** 50 cards labeled by ≥1 annotator (this seed).
- **Better:** 50 cards × 2 independent annotators, then report
  inter-annotator agreement (Cohen's kappa).
- **Best:** 100+ cards × 3 annotators with a consensus protocol.

## What this gets you (vs the old reports/det.json)
- `reports/det.json` evaluates detectors on the *generated pool itself* —
  the data the detectors were used to score in the first place. F1 = 100%
  there is meaningless.
- `reports/holdout_detector_eval.json` (produced by script 37) evaluates
  detectors on a *separate, hand-labeled, off-distribution* set. That F1
  is a real generalization claim you can put in the paper.

## Running the evaluation
After labeling, run:

    python scripts/37_holdout_detector_eval.py \
        --holdout templates/holdout_gpt2_labeled.csv \
        --report_out reports/holdout_detector_eval.json

Read the report; its `detector_metrics.p_ensemble_fake.f1` is the headline
number to put in the paper alongside the JCBlaise test F1.

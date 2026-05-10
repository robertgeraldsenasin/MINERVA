# M.I.N.E.R.V.A. — Changelog

All notable changes to the project. Format inspired by [Keep a Changelog](https://keepachangelog.com).

This project tracks an academic deliverable, not a product release; semantic versioning is approximated rather than strictly followed.

---

## [v2.9.1] — 2026-05-08 — Hotfix

### Fixed
- `tests/test_degnn_graph.py` now uses `pytest.importorskip("torch")` so it skips cleanly when torch isn't installed (Python 3.13 dev environments). Previously it errored at collection time.
- `requirements.txt` now documents Python compatibility tier explicitly (3.10-3.12 supported, 3.13 fallback-only because `feyn` has no 3.13 wheels).

### Added
- `dev-requirements.txt` — lightweight install path (~150 MB) for running the test suite locally without torch/transformers/feyn.

### Notes
- Audit-response code from v2.9.0 unchanged: 231 tests still pass, response bank still 225 phrases at 72% diversity.

---

## [v2.9.0] — 2026-05-08 — Audit-driven refinement

Closes the v2.8.7 audit's three critical findings (P1 #1, P1 #2, P1 #3) plus two important findings (P2 #2, P3 #1).

### Added
- **`scripts/35_pseudonymize_places.py`** — Philippine geographic entity pseudonymizer. Closes P1 #2 (4 PH cities leaked the strict allowlist in v2.8.7). Reads `templates/places_blocklist.txt` (171 entries: 17 regions, 81 provinces, 80+ cities, common landmarks). Writes `generated/cards_pseudo_places.json`. Output report: `reports/pseudo_places.json`.
- **`scripts/37_holdout_detector_eval.py`** — Held-out detector evaluation on hand-labeled GPT-2 cards. Closes P3 #1 (the tautological "100% on the pool" detector metric in v2.8.7's `det.json`). Pure-Python metrics, no sklearn dep.
- **`templates/places_blocklist.txt`** — 171-line hand-curated PH geographic blocklist.
- **`templates/response_bank_v2.json`** — 225 hand-shaped Tagalog phrases across 75 keys, 72% diversity. Schema: `{indicator}/{role}/{tier}` with `phrase_tl`, `phrase_en`, `verifier_action` per entry.
- **`templates/holdout_gpt2_labeled.csv`** — 50-card seed holdout drawn from v2.8.7 GPT-2 outputs (25 fake-target + 25 real-target). `true_label` column intentionally empty for hand-labeling.
- **`templates/holdout_gpt2_labeled.README.md`** — full labeling protocol with single/double/triple-annotator options + Cohen's kappa for the rigorous version.
- **`tests/test_pseudonymize_places.py`** (17 tests including regression for each audit-flagged city).
- **`tests/test_response_bank.py`** (13 tests: schema, coverage, diversity, integration).
- **`tests/test_holdout_eval.py`** (11 tests for metric correctness).
- **`docs/V2.9.0_RELEASE_NOTES.md`** — what changed and why.
- **`docs/V2.9.0_AUDIT_RESPONSE.md`** — point-by-point response to each v2.8.7 audit finding.

### Changed
- **`scripts/29_merge_gpt2_into_pool.py`** — `_build_explanation()` rewritten to consume the v2.9 response bank instead of stub strings ("GPT-2 generation flagged X"). Closes P1 #3 (87.18% faithfulness regression). Added `card_idx` rotation so two cards with same fired indicators get different phrases. NEW pre-merge filter `gpt2_indicators_supported_by_bank()` rejects cards whose fired indicators don't have response-bank coverage.
- **`scripts/10b_prepare_gpt2_neurosymbolic.py`** — report stamps `version: "v2.9.0"` (was `"v2.6.final"` — the visible signal that v2.8.6 binning hadn't activated). Records actual percentile thresholds + `audit_dominant_bin_pct`. Loud warning if dominant bin >70% when `bin_strategy=percentile`.
- **`notebooks/MINERVA_Run_Colab_v2.9.0.ipynb`** — version-bumped throughout. New cell after universal pseudonymize invokes script 35. Balance cell now reads `cards_pseudo_places.json`. New cell after strict allowlist conditionally invokes script 37.

### Test progression
- v2.8.7 baseline: 188 tests
- v2.9.0: **231 tests** (+43, all passing)

---

## [v2.8.7] — 2026-05-05 — GPT-2-to-pool merge

The release that finally produced a non-zero GPT-2 contribution to the final pool.

### Added
- **`scripts/29_merge_gpt2_into_pool.py`** (NEW) — merges GPT-2 cards into the template stream so they reach the pool. Previous pipeline had no merge step; GPT-2 outputs were stranded in jsonl files no script ever read.
- Persistent-generation loop: each label retries with new seeds until `GPT2_MIN_PROMOTED_PER_LABEL` cards survive merge, capped at `GPT2_MAX_ATTEMPTS` to prevent infinite loops.
- Sentence-recovery: trim mid-word truncations back to last `.`, `!`, `?`.
- Excel-style code remap: GPT-2 outputs Excel-style codes (`Candidate AA`, `Candidate DKR`); script 29 remaps to `A/B/C` in order-of-first-appearance.

### Changed
- `max_new_tokens` raised 120 → 200 to reduce truncation upstream.
- 23 new tests; total 188.

### Run results
- Detector RoBERTa F1 (best seed): 95.81% (paper target 95.6% ✓)
- DistilBERT F1 (best seed): 89.96% (paper target 91.0% ✗)
- GPT-2 contribution to final pool: 64/642 cards (10.0%) — first non-zero in v2.8.x series.
- Faithfulness: 87.18% (regression from v2.8.5's 100%; root cause: GPT-2 cards inherited stub explanation block — fixed in v2.9.0).
- Strict allowlist: 96.83% (21 cards rejected, including 4 PH cities — fixed in v2.9.0).

---

## [v2.8.6] — 2026-04-30 — Percentile binning

### Changed
- **`scripts/10b_prepare_gpt2_neurosymbolic.py`** — percentile-based control-token binning. Previous fixed 0.6/0.8 thresholds put 96% of training in 'high' bin; percentile binning targets ~33/33/33 distribution.
- `GPT2_EPOCHS` bumped 3 → 8 (3-epoch run only converged loss 3.49).
- Differentiated seeds for fake/real GPT-2 generation (fake=13, real=27).
- Save cell now bundles `models/qlattice_equation.txt` + `best_qlattice.json`.

### Note (caught in v2.8.7 audit)
This patch did not actually activate in the v2.8.7 run because the user applied v2.8.7 cumulative without v2.8.6 cumulative. v2.9.0 added an audit assertion that catches this regression by stamping `version: "v2.9.0"` in the corpus report.

---

## [v2.8.5] — 2026-04-28 — Dataset bypass

### Fixed
- **`scripts/11b_train_gpt2_neurosymbolic.py`** — replaced broken `load_dataset("text", ...)` with in-memory `Dataset.from_dict({"text": _read_lines(...)})`. Avoids the LocalFileSystem caching incompatibility introduced by datasets ≥ 2.14.

---

## [v2.8.4] — 2026-04-25 — feyn fallback

### Fixed
- **`scripts/08_train_qlattice.py`** — `feyn` pin changed `>=4.0` → `>=3.4,<4.0` (v4 doesn't exist). Added sklearn LogisticRegression fallback that emits a valid `logreg(...)` equation matching the evaluator format. Activated automatically when feyn import fails (e.g. on Python 3.13).

---

## [v2.8.3] — 2026-04-22 — transformers 4.46+ API

### Fixed
- **All training scripts** — migrated from transformers ≤4.45 API to ≥4.46:
  - `evaluation_strategy` → `eval_strategy`
  - `tokenizer=` → `processing_class=`
- `_run_step` rewritten to surface errors past Colab IPython 3.11 traceback bug.

---

## [v2.8.2] — 2026-04-20 — Dataset download fix

### Fixed
- **`scripts/01_download_dataset.py`** — direct ZIP fetch from `https://huggingface.co/datasets/jcblaise/fake_news_filipino/resolve/main/fakenews.zip`. Bypasses the `UnicodeDecodeError 0x8b` that broke v2.7's `load_dataset` path.

---

## [v2.8.0] — 2026-04 — fsspec pin (the v2.7 → v2.8 fix)

### Fixed
- **`requirements.txt`** — pinned `fsspec<=2024.6.1`. Newer versions cause `NotImplementedError: Loading a dataset cached in a LocalFileSystem` which broke v2.7 runs entirely.

---

## [v2.7] — 2026-03 — DE-GNN → RF sequential

### Changed
- DE-GNN now feeds Random Forest as input (was parallel in v2.6). Realizes the SEQUENTIAL pipeline specified in BATB §3.5.2.

---

## [v2.6.final] — 2026-03 — Strict allowlist + neuro-symbolic corpus

### Added
- **`scripts/33_strict_name_allowlist.py`** — final safety-net that rejects any card mentioning unknown names.
- **`scripts/10b_prepare_gpt2_neurosymbolic.py`** — Keskar CTRL-style control-token corpus builder.

---

## Earlier versions (v1.x, v2.0–v2.5)

Predate the cumulative audit work. See git log for details.

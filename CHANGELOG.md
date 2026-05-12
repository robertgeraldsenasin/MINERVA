# M.I.N.E.R.V.A. — Changelog

All notable changes to the project. Format inspired by [Keep a Changelog](https://keepachangelog.com).

This project tracks an academic deliverable, not a product release; semantic versioning is approximated rather than strictly followed.

---

## [v2.9.5] — 2026-05-12 — Audit-driven code improvements

Closes four code-fixable audit findings from the v2.9.4 final-run audit (87/100 composite). Unlike the prior v2.9.4-notebook-polish patch which only fixed markdown headers, v2.9.5 ships actual code changes to surface the audit's diagnostic concerns and remove the lingering Picard 2021 critique.

### Fixed
- **`scripts/11b_train_gpt2_neurosymbolic.py`** — GPT-2 training default seed changed from 42 to **1729** (Hardy-Ramanujan number, used elsewhere in the codebase for consistency). Closes the v2.9.4 audit MEDIUM #3 finding citing Picard (2021): "torch.manual_seed(3407) is all you need" — the seed 42 is over-represented in published ML and creates a cherry-picking risk. Generation seed (script 12b) was already changed to 27 in v2.9.4; training seed now matches.

- **`scripts/35_pseudonymize_places.py`** — report dict `version` field bumped from `"v2.9.0"` to `"v2.9.4"`. Closes v2.9.4 audit LOW #6. Audit-trail consistency: pseudo_places now matches corpus/training/generation reports.

- **`scripts/37_holdout_detector_eval.py`** — same version-stamp bump (`v2.9.0` → `v2.9.4`). Same LOW #6 closure.

- **`scripts/21_balance_unity_cards.py`** — schema-invalid drop now **categorized**, not just counted. Closes v2.9.4 audit MEDIUM #4, which flagged that 414/1314 cards (31.5%) get dropped at the balance stage without explanation. The new `schema_invalid_by_reason` and `schema_invalid_examples_first10` fields in `reports/balance.json` surface the failure modes (missing_required_field, wrong_field_type, invalid_candidate_code, invalid_verdict, invalid_indicator, other) so investigators can prioritize fixes upstream. No change to pass/fail behavior — same cards drop, but for documented reasons.

- **`scripts/32_validate_detectors_on_templates.py`** — explicit `interpretation` and `metric_kind` fields added to `reports/det.json`. Closes v2.9.4 audit MEDIUM #5: the 100% accuracy reported here is mathematically guaranteed because the pool was curated to detector consensus. The new fields explicitly label this as `"metric_kind": "internal_consensus"` (not generalization) and redirect readers to `reports/holdout_detector_eval.json` for the real off-distribution F1. Defensive transparency for panel review.

### Added
- **`tests/test_v295_audit_fixes.py`** (NEW, 12 tests) — regression tests for all five fixes above. Asserts: GPT-2 seed default is 1729 (not 42); scripts 35/37 stamp v2.9.4; script 21 emits the categorized invalid_by_reason fields with all six expected buckets; script 32's interpretation field exists and contains the disambiguating language.

### Test progression
- v2.9.4 baseline: 259 passed
- v2.9.4-notebook-polish: 259 passed (no code changes)
- **v2.9.5: 271 passed** (+12 new tests, 0 regressions)

### Defense impact
Each fix removes a specific panel-question risk:
- Seed=42 question → answered by changing default + citing Picard 2021
- "Why drop 31.5%?" → answered by `schema_invalid_by_reason` breakdown
- "Why 100% on internal eval?" → answered by `metric_kind: internal_consensus` caveat
- Version-stamp consistency → all GPT-2-stage reports now show v2.9.4

### Not addressed in v2.9.5 (operational, not code)
- **Holdout CSV hand-labeling** — still requires ~45 min of human work
- **Pipeline re-run** to capture v2.9.5 numbers in actual output JSONs (~10 min Colab A100)

---

## [v2.9.4-notebook-polish] — 2026-05-12 — Section labeling consistency

Polish-level updates to `notebooks/MINERVA_Run_Colab_v2.9.4.ipynb`. No code or test changes; only markdown section headers corrected for consistency. Test suite still 259 passing.

### Changed (notebook section headers only)
- **Cell 13:** `## 1b. Environment capture` → `## 6b. Environment capture`. The cell sits after section 6 (Working folders), so it should be `6b`, not the misleading `1b`.
- **Cell 33:** `### 7b.6 Held-out test-set evaluation (script 15)` → `### 7b.6 JCBlaise test-set evaluation (script 15)`. Script 15 evaluates on the JCBlaise test SPLIT, not the GPT-2 held-out CSV. "Held-out" caused confusion with section 16b (the real holdout).
- **Cell 58:** `## 16. ... STRICT ALLOWLIST ENFORCER (script 33, NEW v2.6.final)` → drop "NEW v2.6.final" (outdated; script 33 has been in the pipeline since v2.6).
- **Cell 60:** `## 11b. Held-out detector evaluation (v2.9.0)` → `## 16b. Held-out detector evaluation on GPT-2 cards (v2.9.0, script 37)`. The cell sits between section 16 (strict) and section 17 (dashboard); `11b` was clearly wrong.
- **Cell 62:** `## 16. ... Pre-pilot pack` → `## 16c. ... Pre-pilot pack`. Two cells were both numbered `16`; this one renumbered to `16c`.
- **Cell 29:** stale "v2.6.final neuro-symbolic GPT-2 path" reworded to "neuro-symbolic GPT-2 path (introduced in v2.6, refined through v2.9.4)".
- **Cell 38:** stale "the v2.6.final neuro-symbolic conditioning pipeline" reworded to "the neuro-symbolic conditioning pipeline (Mode B, introduced in v2.6, refined through v2.9.4)".

### Preserved
- No code cells modified.
- No tests modified.
- No scripts modified.
- All architectural decisions, hyperparameters, and pipeline ordering unchanged.
- Test suite: 259 passed, 1 skipped (unchanged).

---

## [v2.9.4] — 2026-05-11 — Post-run audit fixes

Closes the three findings from the **professional audit of the v2.9.0 actual Colab run output** (audit dated 2026-05-11). The v2.9.0 audit-response code worked correctly for 4 of 5 audit findings, but the run revealed three additional issues: explanation diversity regressed instead of improving, and two scripts still stamped the wrong version string.

### Fixed
- **`scripts/29_merge_gpt2_into_pool.py`** — explanation summary now rotates through 5 intro variants per label, addressing the P1 #2 regression where v2.9.0 produced only 5.4% explanation diversity (worse than v2.8.7's 8.5%). The bug: `pool.json` measures diversity by counting unique `explanation.summary` strings, but the summary was deterministic given `(target_label, fired_indicators)` — so all cards with the same indicator set got byte-identical summaries. Simulated diversity at 74.4% on 668-card pool (target ≥30%).
- **`scripts/11b_train_gpt2_neurosymbolic.py`** — report dict's `version` field updated from `"v2.6.final"` to `"v2.9.4"`. Cosmetic but appears in audit trail. P2 #3.
- **`scripts/12b_generate_gpt2_neurosymbolic.py`** — same version-string fix. P2 #3.

### Added
- **`tests/test_response_bank.py::test_summary_diversity_across_cards_v294`** — regression test that asserts 5 cards with the same fired-indicator set produce 5 distinct summaries (5 intro variants × 1 indicator-set combo). This locks in the v2.9.4 diversity fix.

### Test progression
- v2.9.3 baseline: 258 tests
- v2.9.4: **259 tests** (+1 regression test, all passing)

### Not addressed in v2.9.4 (deferred to v2.9.5 or run-time tasks)
- Holdout CSV hand-labeling (the 50-card CSV exists; labels are user-side work, not code)
- GPT-2 fine-tune seed changed from 42 to a prime (requires re-running GPT-2 training)
- Ablation script 38 still a scaffold (requires Thesis 3 timeline)

---

## [v2.9.3] — 2026-05-10 — Statistical-validity refinement

Closes the audit findings on experimental discipline: seed count, early stopping, batch-size terminology, and reproducibility reporting.

### Changed
- **`TRAIN_SEEDS` default** bumped from 3 to 5 (`"0,1,2"` → `"13,29,47,89,127"`). Matches RoBERTa paper protocol (Liu et al. 2019: "median over five runs"). Prime numbers chosen to avoid the 42 / cherry-picking risk per Picard 2021. Closes the headline "3 seeds is below standard" finding.
- **`scripts/11b_train_gpt2_neurosymbolic.py`** — added `EarlyStoppingCallback(early_stopping_patience=2)`. Combined with existing `load_best_model_at_end=True`, guarantees the saved model is the best-eval-loss checkpoint, not whatever happened at epoch 8.
- **`scripts/16_train_transformer_classifier.py`** — added `EarlyStoppingCallback(early_stopping_patience=1)`. RoBERTa typically converges by epoch 2 on JCBlaise; this prevents wasted GPU on plateaued training.
- **`scripts/17_run_5seeds_detectors.py`** — emits `reports/detector_seed_stats.json` with mean ± std for both detectors and a paired-sample t-test (RoBERTa vs DistilBERT), n≥3. Also bundles `seed_stats` into the main summary report.
- **Notebook config** — renamed `GPT2_BATCH_SIZE` → `GPT2_GEN_POOL_SIZE` (with back-compat alias). Original name was misleading: it's the persistent-generation pool size per attempt, not a training batch.
- **Notebook config** — GPU-aware batch sizing. Detects A100 / V100 / T4 at config time and sets `(GPT2_TRAIN_BATCH, DETECTOR_BATCH, GRAD_ACCUM)` accordingly. All paths reach effective detector batch=32 (Devlin 2019 standard); A100 runs natively without grad accum, T4 uses (8, 4) as before.

### Added
- **`scripts/minerva_config.py`** (NEW) — centralized hyperparameter module. Single source of truth for SEEDS, EPOCHS, BATCH, LR. Both notebook and scripts import from it. Env-var overrides via `MINERVA_TRAIN_SEEDS`, `MINERVA_GPT2_EPOCHS`, etc. GPU-aware sizing helpers `gpt2_train_batch_for_gpu()` and `detector_batch_for_gpu()`.
- **`scripts/38_ablation_no_conditioning.py`** (NEW) — scaffold for the GPT-2 with-vs-without control-token conditioning ablation. Provides the protocol; running it is Thesis 3 scope.
- **`tests/test_config.py`** (NEW) — 27 tests covering the new config module: defaults match published-paper protocols (Devlin 2019, Liu 2019, Mosbach 2021), env-var overrides, GPU-aware sizing, JSON serialization.
- **Notebook env-capture cell** — dumps GPU name, CUDA version, RAM, derived batch sizes into `reports/_environment.json` for reproducibility.
- **Notebook seed-stats summary cell** — reads `reports/detector_seed_stats.json` and prints panel-ready mean ± std + paired t-test interpretation.
- **`docs/MINERVA_v2.9.3_statistical_validity_audit.html`** — comprehensive audit report with academic citations (Liu 2019, Dodge 2020, Mosbach 2021, Picard 2021, Howard & Ruder 2018, Keskar 2017, Devlin 2019).

### Test progression
- v2.9.2 baseline: 231 tests
- v2.9.3: **258 tests** (+27 from `test_config.py`, all passing)

### References (cited in code comments)
- Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. arXiv:1907.11692
- Dodge, J. et al. (2020). *Fine-Tuning Pretrained Language Models*. arXiv:2002.06305
- Mosbach, M. et al. (2021). *On the Stability of Fine-tuning BERT*. ICLR. arXiv:2006.04884
- Picard, D. (2021). *torch.manual_seed(3407) is all you need*. arXiv:2109.08203
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. arXiv:1810.04805
- Howard, J. & Ruder, S. (2018). *ULMFiT*. ACL. arXiv:1801.06146
- Keskar, N. S. et al. (2017). *On Large-Batch Training for Deep Learning*. ICLR. arXiv:1609.04836

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

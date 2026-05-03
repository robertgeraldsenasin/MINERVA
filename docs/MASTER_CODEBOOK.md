# M.I.N.E.R.V.A. Master Codebook (v2.2)

> *Misinformation Investigation through Networked Embeddings for Rumor Verification and Awareness*
> FEU Institute of Technology — IT Thesis 2026
> Lola, Salva, Senasin

This codebook is the panel-defence-grade reference for every script,
module, and template in the v2.x refactor. For each piece of code we
state:
* **Purpose** — what it does in one sentence.
* **Why** — the design rationale grounded in literature.
* **Inputs / Outputs** — the contract.
* **Pipeline position** — where it sits in the run.
* **Citations** — the literature that justifies the design choice.

Version trail:
- **v2.0**: Original refactor introducing foundation modules and bank-driven explanations.
- **v2.1**: Bug fixes after first run — schema mismatch with script 12, lenient truncation, expanded audit lexicon.
- **v2.2** (current): Post-audit hardening — repetition collapse, off-theme filter expansion, verdict-rule alignment guard, credible-affirmation rewording.

---

## 0. The Five Audit Issues v2.2 Solves

The v2.1 first run produced a structurally complete deck (56 cards, 91% diversity) but five quality issues emerged on inspection. v2.2 addresses each:

| # | Issue | v2.2 Fix |
|---|---|---|
| 1 | Off-theme content (sports/entertainment) — 14% of deck | Patch P2: expanded `ELECTORAL_NEGATIVE` + stronger negative weighting in `keyword_score()` |
| 2 | Legacy "Candidate XX" pseudonym leaks — 45% of deck | Patch P1: wider regex catches all `Candidate <1-3 letters>` patterns |
| 3 | Excessive name repetition — 77% of deck | Patch P1: collapse logic — first mention full name, subsequent mentions short surname |
| 4 | Faithfulness audit at 66% (target ≥95%) | Patch P4: reword credible affirmations + expand audit MISS lexicon |
| 5 | Verdict-rule contradictions (REAL verdict + ≥2 indicators fired) | Patch P3: alignment guard demotes REAL→UNCERTAIN when rules disagree |

All five are documented at length in `docs/V2.2_CHANGES.md`.

---

## 1. Foundation Modules (`scripts/minerva_*.py`)

### 1.1 `minerva_indicators.py` — 12-cue taxonomy

**Purpose.** Provide a deterministic detector for each of 12 student-facing misinformation indicators, plus a flat numeric feature dict for QLattice re-fitting.

**Why.** The legacy pipeline fitted QLattice on 24 PCA components. The discovered equations were of the form `0.62 + 0.41*rpca3 - 0.18*dpca7`. **No SHS student can learn from `rpca3`.** Christensen et al. (2022) and Brolós et al. (2021) document that the interpretability advantage of symbolic regression collapses when its inputs are themselves opaque embeddings.

The 12 indicators are derived from:
* **DEPICT taxonomy** (Roozenbeek & van der Linden, 2019; Basol et al., 2020): Emotion, Polarization, Impersonation, Conspiracy, Discrediting, Trolling.
* **Filipino-electoral extensions** (Arugay & Baquisal, 2022; Schipper, 2025; Bautista, 2021): Historical Revisionism, Manufactured Endorsements, Candidate-Record Fabrication.
* **General misinformation diagnostics** (Leite et al., 2025; W3C Credibility Signals): Anonymous Attribution, Missing Evidence, Urgency, Fabricated Quote.

The result is **12 codes**: `EMO, URG, ANON, MISS, FAB, POL, CONS, DISC, IMP, REV, ENDO, RECF`.

**v2.2 status:** Unchanged. Module is stable.

**Citations.** Roozenbeek & van der Linden (2019); Basol et al. (2020); Khosravi et al. (2022); Longo et al. (2024); Liu, Ye & Li (2024); Athira et al. (2023); Christensen et al. (2022); Brolós et al. (2021); Arugay & Baquisal (2022); Schipper (2025); Bautista (2021); Leite et al. (2025); Cruz et al. (2020 — JCBlaise).

---

### 1.2 `minerva_response_bank.py` — Tiered explanation bank

**Purpose.** Map fired indicators to natural-language feedback phrases, varied by difficulty tier (novice / proficient / advanced) and stamped with a SIFT move (Stop / Investigate / Find / Trace).

**Why.** Two converging requirements forced a bank-driven design over LLM-paraphrased rationales:
1. **Faithfulness preservation.** Longo et al. (2024) and Liu, Ye & Li (2024) both warn that LLM-generated rationales may *plausibly* explain a model's output without *faithfully* tracking it.
2. **Reproducibility for thesis defence.** Same card, same seed → same explanation, every run.

**v2.2 changes:**
- Bank version bumped to `1.1`.
- Four CREDIBLE_AFFIRMATIONS reworded to claim only what the rule layer can verify (i.e. *absence of fired indicators*) instead of unverifiable specifics like "named outlet" or "verifiable date." This was the root cause of audit issue #4.
- Bank IDs updated from `CREDIBLE/v1/*` to `CREDIBLE/v1.1/*`.

**Why this matters:** When a card is REAL by detector score but the underlying text is sports content with no source, claiming "this post names its source" actively lies to the student. The v2.2 phrasing only claims "no misinformation flags fired" — which is true regardless of card topic.

**Citations.** Caulfield (2019); Caulfield & Wineburg (2023); Modirrousta-Galian & Higham (2023); Dehghanzadeh et al. (2024); Almaki et al. (2024); Werner Axelsson (2024); Longo et al. (2024); Liu, Ye & Li (2024).

---

### 1.3 `minerva_candidates.py` — Three fictional candidates

**Purpose.** Maintain a fixed registry of three fictional candidates, plus deterministic archetype-router that maps any post text → one of the three.

**Why three candidates.** Curriculum-design simplicity (the VERIdex module's UI handles three at a time gracefully) and study-grounded selection. Arugay & Baquisal (2022) document three dominant narrative families in 2022 Philippine election disinformation: dynastic restoration, reformist red-tagging, and populist mass-celebrity.

| Code | Name | Archetype | Real-world disinformation pattern |
|---|---|---|---|
| C-RM | Sen. Reynaldo "Rey" Marquez | DYNASTIC | Historical revisionism, manufactured surveys, infrastructure-record inflation |
| C-IB | Vice-Mayor Iris Bantayan | REFORMIST | Red-tagging, fabricated quotes, conspiracy framing |
| C-JS | Rep. Datu Jomar "JM" Salonga | POPULIST | Emotional appeals, unanchored endorsements, celebrity-credibility transfer |

**v2.2 status:** Unchanged. Registry is stable.

**Citations.** Arugay & Baquisal (2022); Mendoza et al. (2022, 2023); Schipper (2025); Bautista (2021); Yermilov et al. (2023); Deinla et al. (2022).

---

### 1.4 `minerva_filters.py` — Four content gates

**Purpose.** Reject cards that fail any of: (a) electoral theme, (b) non-truncation, (c) pseudonym integrity, (d) candidate mention.

**v2.2 changes:**
- `ELECTORAL_NEGATIVE` expanded by 25+ entries to cover sports vocabulary (`boksingero`, `lightweight`, `MGM`, `split decision`, `knockout`, `championship`, `tournament`, `puntos`, etc.) and entertainment vocabulary (`aktres`, `aktor`, `showbiz`, `kapamilya`, etc.).
- `keyword_score()` weighting strengthened: was `pos - 0.6*neg`; now `pos - 1.0*neg` plus `-1.5` penalty when `neg >= pos and neg >= 2`.

**Why this matters.** v2.1's filter accepted any card mentioning a candidate, even if the topic was clearly sports. v2.2 ensures sports-content cards score below threshold even when a candidate is mentioned by name.

**Citations.** Wenzek et al. (2020); Hu et al. (2024 — ARG safety filter).

---

### 1.5 `minerva_schemas.py` — Pydantic v2 contracts

**Purpose.** Type-checked data contracts between scripts.

**v2.2 status:** Unchanged from v2.1.

---

## 2. Refactored Pipeline Scripts

### 2.1 `13_score_generated_with_qlattice.py` (v2.1, preserved in v2.2)

* **Purpose.** Score GPT-2 generations with QLattice + named features + truncation flag.
* **v2.1 fixes:** Reads `p_*_fake` from top-level fields (script 12's actual schema); computes `p_ensemble_fake`; lenient truncation gate; accepts both new and legacy CLIs.
* **v2.2 status:** Unchanged.

### 2.2 `18_verdict_explain.py` (v2.2 has alignment guard added)

* **Purpose.** Convert scored generations into Unity cards with content-aware varied explanations.
* **v2.1 fixes (preserved):** Robust `p_fake` extraction from any of 6 schema variants.
* **v2.2 NEW changes:**
  * **Verdict-rule alignment guard.** If verdict computes to REAL but ≥2 indicators fired, demote to UNCERTAIN with `fake_pct` lifted into the uncertainty band.
  * **`alignment_flag` in provenance.** Records whether verdict and rules agreed.
  * **`pipeline_version` bumped to `2.2.0`.**

### 2.3 `21_balance_unity_cards.py`

* **Purpose.** Schema-validate, dedupe, and balance the unity-cards pool.
* **v2.2 status:** Unchanged.

### 2.4 `22_pseudonymize_entities.py` (v2.2 has the largest patch)

* **Purpose.** Rewrite real-name references to one of the three fictional candidates with session-cache consistency.
* **v2.0 origin:** Replaced random three-letter codes with three fixed candidates.
* **v2.2 NEW changes (the largest patch):**
  * **Wider legacy-pseudonym regex** — catches all `Candidate <1-3 letters>` patterns, not just 3-letter.
  * **Cross-candidate pollution cleanup** — strips fragments from earlier rewrites so collapse doesn't produce Frankenstein names.
  * **Repetition collapse** — first mention full name, subsequent mentions short surname.
  * **Quote-fragment cleanup** — removes orphan `"Rey"`, `"JM"` left when a different candidate's name was substituted.
  * **Verdict-rule alignment guard mirror** — runs the same demotion logic as script 18.

### 2.5 `23_enforce_election_theme.py`

* **Purpose.** Reject off-theme cards; tag neutral-volume cards.
* **v2.2 status:** Code unchanged but behaviour improved by underlying `minerva_filters.py` upgrades.

### 2.6 `24_curate_teaching_cards.py`

* **Purpose.** Promote themed cards into the daily-cycle deck.
* **v2.2 status:** Unchanged.

### 2.7 `25_build_candidate_scenarios.py`

* **Purpose.** Build VERIdex profile cards.
* **v2.2 status:** Unchanged.

### 2.8 `26_faithfulness_audit.py` (v2.2 has expanded MISS lexicon)

* **Purpose.** Re-extract indicators from explanation prose and assert set-equality with `fired_indicators`.
* **v2.1 fix:** Expanded INDICATOR_MENTIONS to match actual bank prose synonyms.
* **v2.2 NEW changes:** Further expansion of MISS lexicon to recognise advanced-tier phrasing (`scores zero`, `named-source`, `external-link`, `credibility-signals`, `W3C`, `Leite et al`).

### 2.9 `27_response_bank_versioning.py`

* **Purpose.** Stamp / diff / re-render decks under different bank versions.
* **v2.2 status:** Unchanged.

---

## 3. Templates

JSON exports of the in-Python registries for Unity client consumption.

---

## 4. Tests

- `tests/test_indicators.py` (24 tests)
- `tests/test_filters.py` (14 tests)

Total: 38 tests passing in v2.2.

---

## 5. Bibliography (selected)

* Roozenbeek, J., & van der Linden, S. (2019). The fake news game. *Journal of Risk Research*.
* Basol, M., Roozenbeek, J., & van der Linden, S. (2020). Good news about Bad News. *Journal of Cognition*.
* Caulfield, M. (2019). SIFT (the four moves). *Hapgood blog*.
* Caulfield, M., & Wineburg, S. (2023). *Verified*. University of Chicago Press.
* Modirrousta-Galian, A., & Higham, P. A. (2023). Conservative response bias in misinformation training. *J. Experimental Psychology: Applied*.
* Arugay, A. A., & Baquisal, J. K. A. (2022). Mobilized and polarized. *Pacific Affairs*.
* Schipper, B. C. (2025). Disinformation tactics in Philippine electoral discourse. *Data & Policy*.
* Mendoza, R. U., et al. (2022, 2023).
* Khosravi, H., et al. (2022). Explainable AI in education. *Computers & Education: AI*.
* Longo, L., et al. (2024). Open problems in explainable AI. *Information Fusion*.
* Liu, Y., Ye, S., & Li, M. (2024). Faithfulness vs plausibility in LLM rationales. *EMNLP*.
* Christensen, T., et al. (2022). Symbolic regression with QLattice. *Discover Computing*.
* Brolós, K., Christensen, T., et al. (2021). QLattice. *arXiv:2104.05417*.
* Yermilov, P., et al. (2023). Consistency-preserving pseudonymisation. *EACL*.
* Cruz, J. C. B., Tan, J. K. C., & Cheng, C. K. (2020). JCBlaise/Filipino corpus.

---

*Last updated: v2.2 release (03 May 2026).*

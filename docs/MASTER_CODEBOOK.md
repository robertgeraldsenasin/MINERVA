# M.I.N.E.R.V.A. Master Codebook (v2.3)

> *Misinformation Investigation through Networked Embeddings for Rumor Verification and Awareness*
> FEU Institute of Technology — IT Thesis 2026
> Lola, Salva, Senasin

This codebook is the panel-defence-grade reference for every script,
module, and template in the v2.x refactor.

Version trail:
- **v2.0**: Original refactor (foundation modules, bank-driven explanations)
- **v2.1**: Bug fixes after first run (schema mismatch, lenient truncation, audit lexicon)
- **v2.2**: Post-audit hardening (repetition collapse, off-theme expansion, alignment guard, credible affirmations rewording)
- **v2.3** (current): Dynamic-content system (500-card pool + per-user deck draw)

---

## 0. The Architecture in Two Sentences

GPT-2 generates 3000 raw posts; QLattice + detectors score them; the explainability layer assigns indicators and produces feedback prose. The result is a **500-card pool** that ships with the Unity APK; at runtime, each Filipino SHS student gets their own **56-card deck deterministically drawn from the pool** by their student ID, so no two students see the same content.

---

## 1. Foundation Modules (`scripts/minerva_*.py`)

### 1.1 `minerva_indicators.py` — 12-cue taxonomy

**Purpose.** Detect 12 student-facing misinformation cues (`EMO, URG, ANON, MISS, FAB, POL, CONS, DISC, IMP, REV, ENDO, RECF`) plus produce a flat numeric feature dict.

**Why these 12.** Combination of:
- DEPICT taxonomy (Roozenbeek & van der Linden 2019; Basol et al. 2020)
- Filipino-electoral extensions (Arugay & Baquisal 2022; Schipper 2025; Bautista 2021)
- General misinformation diagnostics (Leite et al. 2025; W3C Credibility Signals)

**v2.3 status:** Unchanged.

### 1.2 `minerva_response_bank.py` — Tiered explanation bank

**Purpose.** Map fired indicators to natural-language feedback phrases, varied by tier (novice/proficient/advanced) and SIFT move (Stop/Investigate/Find/Trace).

**Why bank-driven instead of LLM-paraphrased.** Faithfulness preservation (Longo et al. 2024; Liu, Ye & Li 2024) and reproducibility — same card same seed always gives same explanation.

**v2.3 status:** Unchanged from v2.2 (bank version 1.1, credible affirmations claim only what rule layer can verify).

### 1.3 `minerva_candidates.py` — Three fictional candidates

**Purpose.** Fixed registry of three candidates plus deterministic archetype router.

| Code | Name | Archetype | Real-world disinformation pattern |
|---|---|---|---|
| C-RM | Sen. Reynaldo "Rey" Marquez | DYNASTIC | Historical revisionism, manufactured surveys |
| C-IB | Vice-Mayor Iris Bantayan | REFORMIST | Red-tagging, fabricated quotes |
| C-JS | Rep. Datu Jomar "JM" Salonga | POPULIST | Emotional appeals, celebrity endorsement |

**Why three.** Curriculum simplicity + study-grounded (Arugay & Baquisal 2022).

**v2.3 status:** Unchanged.

### 1.4 `minerva_filters.py` — Four content gates

**Purpose.** Reject cards that fail theme, truncation, pseudonym, or candidate-mention gates.

**v2.3 status:** Unchanged from v2.2 (`ELECTORAL_NEGATIVE` includes 25+ sports/showbiz terms; `keyword_score` weights negatives at 1.0).

### 1.5 `minerva_schemas.py` — Pydantic v2 contracts

**v2.3 changes:**
- `StoryCard.day` made optional. Pool cards have `day=None`; drawn deck cards have `day=1..7`.
- `StoryCard.pool_index` added (set by curator at v2.3).

---

## 2. Pipeline Scripts

### 2.1 `13_score_generated_with_qlattice.py` (v2.1)

* **Purpose.** Score GPT-2 generations with QLattice + named features + truncation flag.
* **v2.1 fixes:** Reads `p_*_fake` from top-level fields; lenient truncation; legacy CLI compat.
* **v2.3 status:** Unchanged.

### 2.2 `18_verdict_explain.py` (v2.2)

* **Purpose.** Convert scored generations into Unity cards with content-aware explanations.
* **v2.2 changes:** Verdict-rule alignment guard; `alignment_flag` provenance; pipeline_version bumped to 2.2.0.
* **v2.3 status:** Unchanged.

### 2.3 `21_balance_unity_cards.py` (v2.3 — default bumped)

* **Purpose.** Schema-validate, dedupe, and balance the unity-cards pool across verdict × candidate × difficulty.
* **v2.3 change:** Default `--target_total` bumped from 200 → 500. Supports the 500-card pool needed for dynamic content.

### 2.4 `22_pseudonymize_entities.py` (v2.2)

* **Purpose.** Rewrite real-name references to one of the three fictional candidates.
* **v2.2 changes:** Wider regex; cross-candidate pollution cleanup; repetition collapse; quote-fragment cleanup.
* **v2.3 status:** Unchanged.

### 2.5 `23_enforce_election_theme.py` (v2.2)

* **Purpose.** Reject off-theme cards.
* **v2.3 status:** Unchanged (behaviour driven by v2.2 expansion of `ELECTORAL_NEGATIVE`).

### 2.6 `24_curate_teaching_cards.py` (v2.3 — REWRITTEN)

* **Purpose:** v2.3 produces a **POOL of curated cards**, not a single fixed deck. Per-user deck assignment is now done by script 28.
* **v2.3 changes (the largest):**
  * Output renamed: `unity_cards_pool.json` (was `story_cards.json`).
  * Output format: `{"_metadata": {...}, "cards": [...]}` (was flat list).
  * Cards are pool-indexed and day-agnostic (no `day` field).
  * `--target_pool_size` flag (default 500) replaces `--target_total`.
  * Quotas (REAL ratio, candidate spread, indicator coverage) enforced at the pool level.
* **Legacy compat:** Old `--days/--cards_per_day/--min_credible_per_day` flags still accepted but informational only — they're carried into pool metadata for the draw script to use as defaults.

### 2.7 `25_build_candidate_scenarios.py` (v2.3 — pool-aware loader)

* **Purpose.** Build VERIdex profile cards.
* **v2.3 change:** `_load_cards_or_pool()` helper accepts both flat lists and pool dicts so this script works on either.

### 2.8 `26_faithfulness_audit.py` (v2.3 — pool-aware loader)

* **Purpose.** Re-extract indicators from explanation prose and assert set-equality with `fired_indicators`.
* **v2.3 change:** Same loader helper — audit now runs on the entire pool, not a single deck.

### 2.9 `27_response_bank_versioning.py` (v2.3 — pool-aware loader)

* **Purpose.** Stamp / diff / re-render decks under different bank versions.
* **v2.3 change:** Same loader helper.

### 2.10 `28_draw_user_deck.py` (v2.3 — NEW)

* **Purpose.** Deterministically draw a per-user deck from the curated pool.
* **Why this exists:** Enables dynamic content. Each Filipino SHS student gets their own deck drawn by `(user_id, pool_hash)`, so no two students see identical content. Same student replaying gets the same deck.
* **Algorithm:**
  1. Compute SHA-256 of `f"{user_id}:{pool_hash}"`, use first 8 bytes as int64 RNG seed.
  2. For each day 1..7:
     - Reserve REAL quota (≥3/day per Modirrousta-Galian & Higham 2023).
     - Fill candidate-coverage gaps (≥1/candidate/day when supported).
     - Pad remaining slots with FAKE (mostly) and UNCERTAIN (~15%).
     - Shuffle the day so REAL cards aren't all clustered first.
  3. Cross-link each FAKE to the most recent REAL the player has seen (for VERIdict's credible-counter pairing).
* **Output:** `deck_<user_id>.json` with metadata header + cards array.
* **Determinism property:** Two students with different `user_id` get different decks. Same student replaying gets the same deck.
* **Citations:** Modirrousta-Galian & Higham (2023) — credible-card quota; Yermilov et al. (2023) — deterministic pseudonymisation; Christensen et al. (2022) — per-card auditability.

---

## 3. Pipeline Volume (v2.3)

| Stage | Count | Notes |
|---|---|---|
| GPT-2 generation | 3000 raw (1500/class) | Bumped from 1000 in v2.2 |
| Script 13 scoring | ~2400 records | ~80% pass-through |
| Script 21 balance | 500 cards | New default `--target_total 500` |
| Script 22 pseudonymize | 500 cards | No count change |
| Script 23 theme filter | ~450 cards | Some sports rejected |
| Script 24 curate POOL | 500 cards (or as much as available) | NOT a fixed deck anymore |
| Script 28 per-user draw | 56 cards/user | One deck file per student |

---

## 4. Templates

JSON exports of the in-Python registries for Unity client consumption. v2.3 ships:
- `templates/candidate_profiles_three_candidates.json`
- `templates/teaching_response_bank_v1.json`
- `templates/teaching_response_bank_v1_export.json` (bank v1.1)
- `templates/election_theme_keywords.json`
- `templates/indicator_taxonomy_v1.json`

---

## 5. Tests

- `tests/test_indicators.py` (24 tests)
- `tests/test_filters.py` (14 tests)

Total: 38 tests passing in v2.3.

---

## 6. Dynamic Content Capacity

| Pool size | Max non-overlap decks | Pairwise overlap | Use case |
|---|---|---|---|
| 200 | 3 | ~28% | Smoke test only |
| **500 (v2.3 default)** | **8–9** | **~11%** | First SHS pilot (≤50 students) |
| 800 | 14 | ~7% | Mid-scale (≤100 students) |
| 1000+ | 17+ | ~6% | Full thesis evaluation |

---

## 7. Selected Bibliography

* Roozenbeek, J., & van der Linden, S. (2019). The fake news game. *Journal of Risk Research*.
* Basol, M., Roozenbeek, J., & van der Linden, S. (2020). Good news about Bad News. *Journal of Cognition*.
* Caulfield, M. (2019). SIFT (the four moves). *Hapgood blog*.
* Caulfield, M., & Wineburg, S. (2023). *Verified*. University of Chicago Press.
* Modirrousta-Galian, A., & Higham, P. A. (2023). Conservative response bias in misinformation training. *J. Experimental Psychology: Applied*.
* Arugay, A. A., & Baquisal, J. K. A. (2022). Mobilized and polarized. *Pacific Affairs*.
* Schipper, B. C. (2025). Disinformation tactics in Philippine electoral discourse. *Data & Policy*.
* Christensen, T., et al. (2022). Symbolic regression with QLattice. *Discover Computing*.
* Brolós, K., Christensen, T., et al. (2021). QLattice. *arXiv:2104.05417*.
* Yermilov, P., et al. (2023). Consistency-preserving pseudonymisation. *EACL*.
* Cruz, J. C. B., Tan, J. K. C., & Cheng, C. K. (2020). JCBlaise/Filipino corpus.

---

*Last updated: v2.3 release (03 May 2026).*

# M.I.N.E.R.V.A. Master Codebook (v2.4)

> *Misinformation Investigation through Networked Embeddings for Rumor Verification and Awareness*
> FEU Institute of Technology — IT Thesis 2026
> Lola, Salva, Senasin

This codebook is the panel-defence-grade reference for every script,
module, and template in the v2.x refactor.

Version trail:
- **v2.0**: Original refactor (foundation modules, bank-driven explanations)
- **v2.1**: Bug fixes (schema mismatch, lenient truncation, audit lexicon)
- **v2.2**: Post-audit hardening (repetition collapse, off-theme expansion, alignment guard)
- **v2.3**: Dynamic-content system (500-card pool + per-user deck draw)
- **v2.4** (current): Three-candidate enforcement (unified placeholder regex, tier-ratio fix, alignment-threshold tuning)

---

## 0. The Architecture in Three Sentences

GPT-2 generates 3000+ raw posts; QLattice + detectors score them; the explainability layer assigns indicators and produces feedback prose.
The result is a **500-card pool** where every card is routed to one of exactly three fictional candidates (C-RM Marquez, C-IB Bantayan, C-JS Salonga); any GPT-2 placeholder like Entity X or Candidate Y is rewritten to the cue-routed canonical name.
At runtime each Filipino SHS student gets their own **56-card deck deterministically drawn from the pool** by their student ID, so no two students see the same content.

---

## 1. The Three-Candidate Constraint (CRITICAL design choice)

Per thesis §1.5 Scope and Arugay & Baquisal (2022) on Philippine election narrative archetypes, MINERVA uses exactly three fictional candidates. **No other candidate codes, entity codes, or person codes appear in any final card.**

| Code | Name | Archetype | Real-world disinformation pattern |
|---|---|---|---|
| C-RM | Sen. Reynaldo "Rey" Marquez | DYNASTIC | Historical revisionism, manufactured surveys |
| C-IB | Vice-Mayor Iris Bantayan | REFORMIST | Red-tagging, fabricated quotes |
| C-JS | Rep. Datu Jomar "JM" Salonga | POPULIST | Emotional appeals, celebrity endorsement |

**Enforcement (v2.4):** the unified regex `\b(?:Candidate|Entity|Person)\s+[A-Z]{1,3}\b` runs in two places — `minerva_filters.py` (detection) and `22_pseudonymize_entities.py` (rewrite). Every match in either gets routed to one of the three canonical candidates by `pseudonymize()`'s archetype router (cue-based, deterministic, session-cached).

---

## 2. Foundation Modules (`scripts/minerva_*.py`)

### 2.1 `minerva_indicators.py` — 12-cue taxonomy

**Purpose.** Detect 12 student-facing misinformation cues (`EMO, URG, ANON, MISS, FAB, POL, CONS, DISC, IMP, REV, ENDO, RECF`) plus produce a flat numeric feature dict.

**v2.4 status:** Unchanged.

### 2.2 `minerva_response_bank.py` — Tiered explanation bank

**Purpose.** Map fired indicators to natural-language feedback phrases, varied by tier (novice/proficient/advanced) and SIFT move (Stop/Investigate/Find/Trace).

**v2.4 changes:**
- `tier_for_card_index()` now takes optional `total_in_session` parameter for **proportional** tier ratio (40% novice, 35% proficient, 25% advanced).
- Legacy absolute-threshold mode preserved when `total_in_session=None` for backward compat.

**Why this matters.** v2.3's audit found 91% of pool cards tagged advanced because the absolute thresholds (idx<10 novice, idx<25 proficient, else advanced) make every card with idx≥25 advanced — and after pseudonymization renumbers to a 64-card pool, almost everything sits past idx=25. Proportional ratio fixes this.

### 2.3 `minerva_candidates.py` — Three fictional candidates

**Purpose.** Fixed registry of three candidates plus deterministic archetype router.

**v2.4 status:** Registry unchanged. The router is now backed by a stronger placeholder regex (P1) that ensures more cards reach it instead of being rejected by the theme filter.

### 2.4 `minerva_filters.py` — Four content gates

**Purpose.** Reject cards that fail theme, truncation, pseudonym, or candidate-mention gates.

**v2.4 changes:**
- `_LEGACY_PSEUDONYM_RE` unified to `\b(?:Candidate|Entity|Person)\s+[A-Z]{1,3}\b`. Catches Entity D-V, Candidate W-AW, Person F — all placeholder families GPT-2 emits.

### 2.5 `minerva_schemas.py` — Pydantic v2 contracts

**v2.4 status:** Unchanged from v2.3.

---

## 3. Pipeline Scripts

### 3.1 `13_score_generated_with_qlattice.py` (v2.1)

* **Purpose.** Score GPT-2 generations with QLattice + named features + truncation flag.
* **v2.4 status:** Unchanged from v2.1.

### 3.2 `18_verdict_explain.py` (v2.4 — alignment threshold raised)

* **Purpose.** Convert scored generations into Unity cards with content-aware explanations.
* **v2.4 changes:** Verdict-rule alignment guard threshold raised from `≥2` to `≥3` fired indicators. Restores REAL-card supply that was being over-aggressively demoted.

### 3.3 `21_balance_unity_cards.py` (v2.3)

* **Purpose.** Schema-validate, dedupe, balance.
* **v2.4 status:** Unchanged from v2.3 (default `--target_total 500`).

### 3.4 `22_pseudonymize_entities.py` (v2.4 — multiple changes)

* **Purpose.** Rewrite real-name references to one of the three fictional candidates.
* **v2.4 changes:**
  * Unified placeholder regex (matches `minerva_filters.py`).
  * Alignment guard threshold raised from ≥2 to ≥3.
  * `tier_for_card_index()` called with `total_in_session=len(cards)` for proportional 40/35/25 tier ratio.

### 3.5 `23_enforce_election_theme.py`

* **Purpose.** Reject off-theme cards.
* **v2.4 status:** Unchanged. **Behaviour drastically improved** because `minerva_filters.py`'s regex now properly delimits "tainted" cards — most v2.3 rejections were spurious.

### 3.6 `24_curate_teaching_cards.py` (v2.3 — pool builder)

* **Purpose.** Build a 500-card pool from themed cards.
* **v2.4 status:** Unchanged from v2.3.

### 3.7 `25_build_candidate_scenarios.py` (v2.3)

* **Purpose.** Build VERIdex profile cards.
* **v2.4 status:** Unchanged.

### 3.8 `26_faithfulness_audit.py` (v2.3 + v2.2 lexicon)

* **Purpose.** Re-extract indicators from explanation prose; assert match.
* **v2.4 status:** Unchanged.

### 3.9 `27_response_bank_versioning.py`

* **v2.4 status:** Unchanged.

### 3.10 `28_draw_user_deck.py` (v2.3 — per-user draw)

* **Purpose.** Deterministically draw a per-user deck.
* **v2.4 status:** Unchanged. The draw quality improves automatically once the pool is properly populated by upstream fixes.

---

## 4. Pipeline Volume (v2.4)

| Stage | Count | Notes |
|---|---|---|
| GPT-2 generation | 4000 raw (1500 fake + 2500 real) | REAL bumped from 1500 in v2.4 |
| Script 13 scoring | ~3200 records | ~80% pass-through |
| Script 21 balance | 500 cards | Default `--target_total 500` |
| Script 22 pseudonymize | 500 cards | All Entity/Candidate/Person placeholders rewritten |
| Script 23 theme filter | ~480 cards | Previously rejected ~125 cards spuriously, now keeps them |
| Script 24 curate POOL | 500 cards | Day-agnostic, shipped with APK |
| Script 28 per-user draw | 56 cards/user | One file per student |

---

## 5. Templates

JSON exports for Unity client consumption:
- `templates/candidate_profiles_three_candidates.json`
- `templates/teaching_response_bank_v1.json`
- `templates/teaching_response_bank_v1_export.json` (bank v1.1)
- `templates/election_theme_keywords.json`
- `templates/indicator_taxonomy_v1.json`

---

## 6. Tests

- `tests/test_indicators.py` (24 tests)
- `tests/test_filters.py` (14 tests)

Total: 38 tests passing in v2.4.

---

## 7. Dynamic Content Capacity

| Pool size | Max non-overlap decks | Pairwise overlap | Use case |
|---|---|---|---|
| 64 (v2.3 broken) | 1 | ~90% | Failed |
| ~190 (v2.4 quick fix, no re-gen) | 3 | ~30% | OK for ≤10 students |
| 500 (v2.4 default) | 8–9 | ~11% | First SHS pilot (≤50 students) |
| 1000+ | 17+ | ~6% | Full thesis evaluation |

---

## 8. Selected Bibliography

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

*Last updated: v2.4 release (03 May 2026).*

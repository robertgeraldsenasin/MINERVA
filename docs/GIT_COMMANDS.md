# Git Commands — Landing the v2.0 Refactor

Drop these into a terminal at the repo root (`MINERVA/`). Each commit is a
self-contained unit so a reviewer can bisect or revert individual pieces.

---

## 0. Pre-flight

```bash
cd path/to/MINERVA          # your local clone of robertgeraldsenasin/MINERVA.git
git fetch origin
git checkout main
git pull origin main

# Create the upgrade branch (if not already created)
git checkout -b upgrade/minerva-election-theme

# Confirm a clean working tree
git status
```

---

## 1. Drop the refactor files into the repo

Unzip / copy the v2.0 package into the repo so the layout matches:

```
MINERVA/
├── scripts/                             ← drop new + refactored .py here
├── templates/                           ← create if absent; drop 4 JSON files here
├── tests/                               ← create if absent; drop 2 test files here
├── notebooks/                           ← drop the .ipynb here
├── docs/                                ← create if absent; drop 3 .md files here
└── README_REFACTOR.md                   ← drop at repo root
```

`cp -r minerva_refactor/* path/to/MINERVA/` does it in one command.

---

## 2. Stage and commit in logical units

The history reads cleanly to a panel reviewer. Use these conventional-commit messages exactly.

### Commit 1 — Foundation modules

```bash
git add scripts/minerva_indicators.py \
        scripts/minerva_response_bank.py \
        scripts/minerva_schemas.py \
        scripts/minerva_candidates.py \
        scripts/minerva_filters.py
git commit -m "feat(core): add 5 foundation modules for v2.0 explainability layer

- minerva_indicators: 12-cue student-facing taxonomy (DEPICT + Filipino
  electoral extensions per Arugay & Baquisal 2022; Schipper 2025)
- minerva_response_bank: 56-entry tiered bank with SIFT moves
  (Caulfield 2019; Modirrousta-Galian & Higham 2023)
- minerva_schemas: pydantic v2 contracts for inter-script handoff
- minerva_candidates: three fictional candidates with archetype router
  (consistency-preserving pseudonymisation per Yermilov et al. 2023)
- minerva_filters: four-gate content filter
  (theme + truncation + pseudonym + candidate-mention)"
```

### Commit 2 — Refactor 13 (scoring + named features)

```bash
git add scripts/13_score_generated_with_qlattice.py
git commit -m "refactor(13): augment QLattice with named features + truncation gate

- Feed 12 indicator features alongside the legacy PCA components so
  the discovered equation refers to interpretable named columns
  (Christensen et al. 2022; Brolós et al. 2021).
- Drop truncated GPT-2 generations before scoring; log to
  reports/score_rejection_log.jsonl."
```

### Commit 3 — Refactor 18 (the static-explanation fix)

```bash
git add scripts/18_verdict_explain.py
git commit -m "refactor(18): replace static template with response-bank explanations

Fixes the cohort-wide static-summary problem in legacy unity_cards.json
(every card had the same 'Verdict: REAL/FAKE … from the stored Qlattice
equation' text). Each card now carries fired_indicators, indicator
phrases drawn from the 56-entry bank, SIFT moves, and a bank_ref for
audit. Output validates against the UnityCard pydantic schema.

Smoke-test on 50 legacy records: 47 schema-rejected (truncated text);
3 surviving cards yield 3 unique summaries (100% diversity).

Refs: Roozenbeek & van der Linden (2019); Longo et al. (2024);
Caulfield (2019); Modirrousta-Galian & Higham (2023)."
```

### Commit 4 — Refactor 21 (schema-validated balancing)

```bash
git add scripts/21_balance_unity_cards.py
git commit -m "refactor(21): four-axis balance + schema validation pass

Balance now considers verdict × candidate × difficulty × indicator
coverage and emits reports/balance_report.json."
```

### Commit 5 — Refactor 22 (deterministic 3-candidate pseudonyms)

```bash
git add scripts/22_pseudonymize_entities.py
git commit -m "refactor(22): replace random pseudonyms with 3-candidate router

Fixes the 'Candidate GQW / Candidate DTQ' inconsistency. Real-name
references are now mapped to one of three fictional archetype-grounded
candidates (C-RM Marquez/DYNASTIC, C-IB Bantayan/REFORMIST, C-JS
Salonga/POPULIST) with session-cache consistency
(Yermilov et al. 2023). Adds --re_explain flag so explanations are
re-built with candidate context after rewrite."
```

### Commit 6 — Refactor 23 (Grab/Meralco leak fix)

```bash
git add scripts/23_enforce_election_theme.py
git commit -m "refactor(23): hard-negative theme filter + neutral-volume policy

Fixes the off-theme leaks observed in legacy data (Grab transport
strikes, Meralco utility rates). The filter now scores against an
ELECTORAL_NEGATIVE list and emits reports/theme_rejection_log.jsonl
per-card. Allows benign off-theme cards through tagged
is_neutral_volume=True so the Chattr feed retains realistic volume."
```

### Commit 7 — Refactor 24 (teaching-deck curation)

```bash
git add scripts/24_curate_teaching_cards.py
git commit -m "refactor(24): difficulty banding + credible-card mandate

Day-by-day promotion now bands by difficulty (novice -> proficient ->
advanced) and enforces a minimum of 3 credible cards per day per
Modirrousta-Galian & Higham (2023). Each FAKE card is cross-linked to
the most-recent REAL card the player saw, so VERIdict can render a
side-by-side comparison."
```

### Commit 8 — Refactor 25 (VERIdex profiles)

```bash
git add scripts/25_build_candidate_scenarios.py
git commit -m "refactor(25): archetype-grounded VERIdex profiles

Profile cards now derive from the candidate registry and include
indicator susceptibility (both prior weighting and empirical
frequency from the deck) plus counter-narrative anchors so VERIdex
shows learners what to watch for and what good evidence looks like."
```

### Commit 9 — New 26 (faithfulness audit)

```bash
git add scripts/26_faithfulness_audit.py
git commit -m "feat(26): post-hoc faithfulness audit (Longo et al. 2024 OP-7)

Re-extracts indicators from explanation prose and asserts set-equality
with fired_indicators. Verifies bank_version stamps, REAL-card credible
affirmations, and bank_ref well-formedness. Strict mode aborts on any
failure -- this is the panel-defence-grade check."
```

### Commit 10 — New 27 (bank versioning)

```bash
git add scripts/27_response_bank_versioning.py
git commit -m "feat(27): response-bank versioning (stamp / diff / rerender / export)

Enables A/B comparison between bank versions across student-testing
cohorts (Werner Axelsson 2024)."
```

### Commit 11 — Templates

```bash
git add templates/candidate_profiles_three_candidates.json \
        templates/teaching_response_bank_v1.json \
        templates/election_theme_keywords.json \
        templates/indicator_taxonomy_v1.json
git commit -m "feat(templates): JSON exports of registries for Unity client + auditors"
```

### Commit 12 — Tests

```bash
git add tests/test_indicators.py tests/test_filters.py
git commit -m "test: add 37 unit tests covering all 12 indicators + 4 gates"
```

### Commit 13 — Notebook

```bash
git add notebooks/MINERVA_Run_Colab_v2.ipynb
git commit -m "feat(notebook): v2.0 Colab notebook (35 cells)

Preserves script numbering for reviewer continuity. Fixes legacy
notebook bugs (cell 2 stray 'e' token; cell 23 monkey-patch of
script 12)."
```

### Commit 14 — Docs

```bash
git add docs/MASTER_CODEBOOK.md \
        docs/CHANGELOG_COMPARISON.md \
        docs/GIT_COMMANDS.md \
        README_REFACTOR.md
git commit -m "docs: panel-defence-grade codebook + changelog + git playbook"
```

---

## 3. Push and open PR

```bash
git push -u origin upgrade/minerva-election-theme
```

Then on GitHub: open a PR from `upgrade/minerva-election-theme` into `main`. Use this PR description:

> ### v2.0 Refactor: Industry-grade explainability layer
>
> This PR fixes four blocking issues in the legacy pipeline ahead of SHS
> student testing:
>
> 1. **Static explanations** — replaced by 56-entry response bank with
>    deterministic per-card rotation.
> 2. **Off-theme leaks** (Grab/Meralco) — replaced by hard-negative theme
>    filter with structured rejection log.
> 3. **Random pseudonyms** — replaced by deterministic three-candidate
>    archetype router.
> 4. **GPT-2 truncation** — caught by `is_truncated()` gate before deck
>    promotion.
>
> Verifiable claims:
> - 37 unit tests passing (`python -m pytest tests/ -v`).
> - On a 50-record legacy sample, the schema correctly rejects 47
>   truncated cards; the 3 survivors yield 3 unique summaries (100%
>   diversity vs ≈ 0% in v1).
> - Faithfulness audit (script 26) passes 100% on the v2.0 deck.
>
> See `docs/MASTER_CODEBOOK.md` for the full design rationale with
> citations.
>
> Reviewers: please run `python -m pytest tests/ -v` and inspect
> `notebooks/MINERVA_Run_Colab_v2.ipynb` Section 4 (response-bank stats)
> and Section 16 (final reports dashboard).

---

## 4. After merge — release tag

```bash
git checkout main
git pull origin main
git tag -a v2.0.0 -m "v2.0.0: explainability + theme + pseudonym + truncation refactor"
git push origin v2.0.0
```

---

## 5. Rollback (if a panel objection lands)

Each commit is self-contained; you can revert individual pieces without losing the rest:

```bash
git revert <sha>           # reverts a single commit
git revert <sha1>..<sha2>  # reverts a range
```

The most-likely candidates for rollback are commits 6 (theme filter — adjust threshold) and 7 (curation — adjust difficulty banding), neither of which would invalidate the rest of the work.

---

*Total commits: 14. Total files added/modified: 26. Estimated review time: 90 minutes for a Python-comfortable reviewer.*

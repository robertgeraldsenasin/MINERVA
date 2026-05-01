# M.I.N.E.R.V.A. v2.0 Refactor — Drop-In Package

> **Misinformation Investigation through Networked Embeddings for Rumor Verification and Awareness**
> FEU Institute of Technology IT thesis 2026 — Lola, Salva, Senasin
> Repo: `https://github.com/robertgeraldsenasin/MINERVA.git`
> Target branch: `upgrade/minerva-election-theme`

This package replaces the legacy v1 explainability layer with an industry-grade v2.0 layer that fixes four blocking issues identified ahead of SHS student testing. **No model retraining is required.** The refactor operates entirely on the post-detection / explanation / curation stages.

---

## What This Fixes

1. **Static explanations** — every legacy card had the same summary boilerplate ("*Verdict: REAL/FAKE … from the stored Qlattice equation*"). v2.0 replaces this with a 56-entry response bank yielding ~100% per-card diversity.
2. **Off-theme leaks** — Grab/Meralco/transport posts slipped through the legacy filter. v2.0 adds a hard-negative aware classifier + structured rejection log.
3. **Random pseudonyms** — legacy emitted "Candidate GQW", "Candidate DTQ", "Entity B". v2.0 replaces with three fixed fictional candidates with archetype-grounded routing.
4. **Truncated GPT-2 generations** — silently shipped to deck in v1. v2.0 catches them at the schema validator and the `is_truncated()` gate.

---

## Quick-Start (5 minutes)

```bash
# 1. From repo root
git checkout -b upgrade/minerva-election-theme

# 2. Drop this package in
cp -r minerva_refactor/* path/to/MINERVA/

# 3. Install the one new dependency
pip install pydantic

# 4. Run the unit tests
python -m pytest tests/ -v
# Expect: 37 passed

# 5. Run the v2 Colab notebook end-to-end
# Open notebooks/MINERVA_Run_Colab_v2.ipynb in Colab and Run All
```

---

## What's in This Folder

```
minerva_refactor/
├── README_REFACTOR.md              ← you are here
├── scripts/                        (15 Python modules / scripts)
│   ├── minerva_indicators.py       (12-cue student-facing taxonomy)
│   ├── minerva_response_bank.py    (56-entry tiered feedback bank)
│   ├── minerva_schemas.py          (pydantic v2 contracts)
│   ├── minerva_candidates.py       (3 fictional candidates + archetype router)
│   ├── minerva_filters.py          (4 content gates)
│   ├── 13_score_generated_with_qlattice.py  (REFACTORED)
│   ├── 18_verdict_explain.py                (REFACTORED — static-explanation fix)
│   ├── 21_balance_unity_cards.py            (REFACTORED)
│   ├── 22_pseudonymize_entities.py          (REFACTORED — 3-candidate fix)
│   ├── 23_enforce_election_theme.py         (REFACTORED — Grab/Meralco fix)
│   ├── 24_curate_teaching_cards.py          (REFACTORED)
│   ├── 25_build_candidate_scenarios.py      (REFACTORED)
│   ├── 26_faithfulness_audit.py             (NEW — panel-defence audit)
│   └── 27_response_bank_versioning.py       (NEW — A/B versioning)
├── templates/                      (4 JSON exports for Unity client + auditors)
│   ├── candidate_profiles_three_candidates.json
│   ├── teaching_response_bank_v1.json
│   ├── election_theme_keywords.json
│   └── indicator_taxonomy_v1.json
├── tests/                          (37 pytest unit tests)
│   ├── test_indicators.py
│   └── test_filters.py
├── notebooks/
│   └── MINERVA_Run_Colab_v2.ipynb  (35-cell drop-in replacement)
└── docs/                           (panel-defence-grade documentation)
    ├── MASTER_CODEBOOK.md          ← every script + design decision + citation
    ├── CHANGELOG_COMPARISON.md     ← v1 vs v2 metric-by-metric
    └── GIT_COMMANDS.md             ← exact 14-commit landing playbook
```

---

## What Stays the Same

* GPT-2 fine-tuned on JCBlaise — unchanged.
* RoBERTa-Filipino detector head — unchanged.
* DistilBERT-multilingual detector head — unchanged.
* DE-GNN — unchanged.
* Random Forest ensemble — unchanged.
* QLattice symbolic regression — same model, **augmented inputs** (now sees 12 named indicators alongside the legacy PCA components).

This is a deliberately surgical refactor: it touches only the explanation, curation, and pseudonymisation stages so the team can ship to student testing without re-training anything.

---

## Verifiable Claims for the Panel

| Claim | How to verify |
| ----- | ------------- |
| 37 unit tests pass | `python -m pytest tests/ -v` |
| Static-explanation fix works on legacy data | Re-run script 18 on the legacy `unity_cards.json` (script generates `audit_18.jsonl`); the 3 surviving cards yield 3 unique summaries vs the legacy 1-template behaviour. |
| Grab/Meralco leak is blocked | `python scripts/minerva_filters.py` — sample test prints the rejection. |
| 3-candidate routing is consistent | `python scripts/minerva_candidates.py` — same name → same code across re-runs. |
| Faithfulness audit passes | `python scripts/26_faithfulness_audit.py --in_file generated/story_cards.json --strict` (must exit 0). |
| Bank version + hash stamped on every card | Inspect any card's `provenance.bank_version` and `provenance.bank_hash`. |

---

## Citations (full list in `docs/MASTER_CODEBOOK.md`)

The design decisions are grounded in published literature: Roozenbeek & van der Linden (2019) for the Bad News taxonomy that became the indicator scheme; Caulfield (2019) for the SIFT moves at the end of every bank entry; Modirrousta-Galian & Higham (2023) for the credible-card mandate that prevents over-suspicion drift; Arugay & Baquisal (2022), Schipper (2025), and Mendoza et al. (2022, 2023) for the Filipino-electoral archetype grounding; Longo et al. (2024) and Liu, Ye & Li (2024) for the faithfulness contract that script 26 audits; Christensen et al. (2022) and Brolós et al. (2021) for the named-feature-over-PCA argument that drives script 13.

---

## Where to Go Next

* **First-time onboarding:** read `docs/MASTER_CODEBOOK.md` end-to-end (~30 minutes).
* **Landing the PR:** follow `docs/GIT_COMMANDS.md` step-by-step (14 commits, ~30 minutes).
* **Running it:** open `notebooks/MINERVA_Run_Colab_v2.ipynb` in Colab.
* **Defence prep:** print `docs/CHANGELOG_COMPARISON.md` for the panel — it is the single best one-page summary of what changed.

---

*v2.0 release. For Term-3 student testing. The next planned bank version (v1.1) will add Bisaya code-switching variants, scheduled after the first round of student feedback.*

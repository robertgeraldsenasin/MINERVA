# M.I.N.E.R.V.A.

**M**isinformation **I**nspector for **N**ew **E**lection-cycle **R**isks via **V**erifiable **A**rtifacts.

An Android Unity educational game that teaches Senior High School Filipino voters to recognize political-candidate misinformation through psychological inoculation (in the tradition of Cambridge's [Bad News](https://www.getbadnews.com/) and [Harmony Square](https://www.harmonysquare.game/)). FEU IT undergraduate thesis.

**Status:** v2.6-final-consolidated + P1.2 (pre-pilot pack) — 668-card pool, 12/12 DEPICT indicator coverage, 100% faithfulness audit, 39/39 unit tests passing.

---

## What's in this repo

```
MINERVA/
├── CLAUDE.md             # Briefing for Claude Code agents
├── HANDOFF.md            # Open work items (P1.x defense prep, P2.x robustness, etc.)
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── .gitignore
├── scripts/              # 40+ Python scripts (numbered pipeline stages)
│   ├── candidate_config.py            # ⭐ THE editable config (3 candidates)
│   ├── 30_template_scenario_generator.py     # ⭐ 18 templates
│   ├── 31_universal_pseudonymize.py
│   ├── 32_validate_detectors_on_templates.py # NEW v2.6-final-consolidated
│   ├── 40_export_pilot_pack.py               # NEW P1.2
│   ├── minerva_candidates.py                 # REGISTRY (rebuilds from candidate_config)
│   ├── minerva_filters.py
│   ├── minerva_schemas.py
│   └── ... (detector training scripts 01-19, pipeline scripts 21-32)
├── tests/                # 39 pytest tests
├── docs/
│   ├── V2.6_CHANGES.md           # Full architecture history v2.0 → consolidated
│   ├── MASTER_CODEBOOK.md
│   └── MINERVA_v2.6_Audit.html   # Visual audit (open in browser)
└── notebooks/
    └── MINERVA_Run_Colab_v2.6.ipynb  # ⭐ Colab pipeline runner
```

---

## Quick start (Google Colab)

This is the path most users will take.

1. Open https://colab.research.google.com
2. **File → Upload notebook** → upload `notebooks/MINERVA_Run_Colab_v2.6.ipynb`
3. **Runtime → Run all** (or step through the cells)

Total runtime: **~3 minutes** for the card-generation pipeline (no GPU needed).

The notebook will:
- Mount Google Drive (optional, for saving outputs)
- Clone this repo at the `upgrade/minerva-election-theme` branch
- Install dependencies from `requirements.txt`
- Run all 39 unit tests
- Run the full pipeline: `30 → 31 → 21 → 23 → 24 → 28 → 26 → 32 → 40`
- Print the verified metrics dashboard
- Sample 6 cards from a player's deck
- Export the pre-pilot pack (HTML + questionnaire + answer key)

---

## Quick start (local machine — Windows / WSL / Linux)

```bash
# Clone
git clone https://github.com/robertgeraldsenasin/MINERVA.git
cd MINERVA
git checkout upgrade/minerva-election-theme

# Install
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Test
python -m pytest tests/ -q
# Expected: 39 passed

# Run the pipeline (creates generated/ and reports/)
mkdir -p generated reports/decks
python scripts/30_template_scenario_generator.py --out_file generated/template_cards.json --n_per_template 18 --report_out reports/template_gen.json
python scripts/31_universal_pseudonymize.py --in_file generated/template_cards.json --out_file generated/cards_pseudo.json --report_out reports/pseudo.json
python scripts/21_balance_unity_cards.py --in_file generated/cards_pseudo.json --out_file generated/balanced.json --target_total 700 --report_out reports/balance.json
python scripts/23_enforce_election_theme.py --in_file generated/balanced.json --out_file generated/themed.json --report_out reports/theme.json --rejection_log reports/theme_rej.jsonl
python scripts/24_curate_teaching_cards.py --in_file generated/themed.json --out_file generated/unity_cards_pool.json --reject_out generated/pool_rej.json --report_out reports/pool.json --target_pool_size 700 --days 7 --cards_per_day 8 --min_credible_per_day 3 --seed 1729
python scripts/28_draw_user_deck.py --pool_file generated/unity_cards_pool.json --out_dir generated/decks --user_ids "alice,bob,charlie,diana,erika,fiona,greg,hana" --report_out reports/draw.json
python scripts/26_faithfulness_audit.py --in_file generated/unity_cards_pool.json --report_out reports/faith.json
python scripts/32_validate_detectors_on_templates.py --pool_file generated/unity_cards_pool.json --report_out reports/det.json --markdown_out reports/det.md
python scripts/40_export_pilot_pack.py --pool_file generated/unity_cards_pool.json --out_dir reports/pilot_pack
```

---

## Verified metrics (v2.6-final-consolidated + P1.2)

```
Pool size            : 668 cards
  REAL/FAKE/UNCERTAIN : 108/500/60
  Candidates          : C-A 229, C-B 216, C-C 223
  Tier dist (n/p/a)   : 240/212/216
Templates             : 18 (12/12 DEPICT indicator coverage)
Faithfulness audit    : 100% (668/668)
Pairwise overlap      : 11.48% mean (target <15%; supports 11 non-overlapping decks)
Detector validation   : 100% across RoBERTa, DistilBERT, DE-GNN, ensemble
Unit tests            : 39/39 passing
Pre-pilot pack        : 30 cards, 17/17 manipulation tactics, seed-locked reproducible
```

Detector accuracy on JCBlaise test split (frozen models): RoBERTa 95.6% F1, DistilBERT 91.0% F1, DE-GNN 95.8% F1.

---

## Architecture in one paragraph

**Detection stack (frozen):** RoBERTa-Tagalog + DistilBERT-multilingual + DE-GNN ensemble, with Random Forest and QLattice for symbolic explainability. Trained on the JCBlaise Tagalog news corpus.

**Card-generation pipeline (active):** Template-based "Rule-Constrained Content Generation" (per thesis §1.3 p.12) using 18 documented Filipino electoral disinformation tactics. Three editable candidates with archetypes from Arugay & Baquisal (2022): DYNASTIC, REFORMIST, POPULIST. Common Filipino surnames per Santos & del Rosario (2014) PSA naming-frequency analysis, deliberately excluding surnames tied to current political families.

**Verdict tiers (Modirrousta-Galian & Higham 2023):** novice / proficient / advanced — controls explanation depth shown to the player.

**SIFT moves (Caulfield):** STOP / INVESTIGATE / FIND / TRACE — the player's response options after each card.

---

## The editable config — `scripts/candidate_config.py`

Three candidates live in **one file**. Edit there; the pipeline picks up the change automatically across templates, pseudonymizer, theme filter, schema validator, and balance script.

```python
CANDIDATES_CONFIG = [
    {"code": "C-A", "archetype": "DYNASTIC", "title": "Sen.",
     "first_name": "Ramon", "nickname": "Mon", "last_name": "Cruz", ...},
    {"code": "C-B", "archetype": "REFORMIST", "title": "Vice-Mayor",
     "first_name": "Liza", "nickname": "", "last_name": "Reyes", ...},
    {"code": "C-C", "archetype": "POPULIST", "title": "Rep.",
     "first_name": "Joel", "nickname": "Joel", "last_name": "Garcia", ...},
]
```

Constraints (enforced at import):
- Exactly 3 candidates
- Archetypes must be `{DYNASTIC, REFORMIST, POPULIST}`
- Codes must match `^C-[A-Z0-9\-]{1,8}$`
- Avoid surnames tied to current PH political families (per thesis §1.5 Limitation #2)

Backed by Roozenbeek & van der Linden (2019, 2020) on "fictional examples throughout the game" and Hainmueller, Hangartner, & Yamamoto (2015) PNAS vignette-experiment standard.

---

## The 18 disinformation tactics

| # | Tactic | Verdict | Tier | DEPICT | Source |
|---|---|---|---|---|---|
| 1 | historical_revisionism | FAKE | advanced | MISS+REV | Schipper 2025 |
| 2 | historical_revisionism_truth | REAL | novice | (none) | (REAL counterpart) |
| 3 | red_tagging | FAKE | proficient | EMO+FAB+ANON | Arugay 2022 |
| 4 | fake_celebrity_endorsement | FAKE | novice | EMO+ENDO+FAB | Ong & Cabañes 2018 |
| 5 | urgency_sharing | FAKE | novice | URG+MISS+EMO | Roozenbeek 2019 |
| 6 | fake_survey | FAKE | proficient | ENDO+FAB+MISS | Roozenbeek 2019 |
| 7 | credible_policy_announcement | REAL | novice | (none) | Modirrousta-Galian 2023 |
| 8 | ambiguous_allegation | UNCERTAIN | advanced | ANON+MISS | Schafer 2024 |
| 9 | conspiracy_theory | FAKE | advanced | CONS+ANON+MISS | Roozenbeek 2019 (DEPICT) |
| 10 | polarization_us_vs_them | FAKE | proficient | POL+EMO+MISS | Roozenbeek 2020 |
| 11 | discrediting_personal_attack | FAKE | novice | DISC+EMO+ANON | Roozenbeek 2019 |
| 12 | fake_account_impersonation | FAKE | advanced | IMP+FAB+ANON | Ong & Cabañes 2018 |
| 13 | recycled_old_content | FAKE | novice | RECF+MISS+EMO | Schipper 2025 |
| 14 | deepfake_video_claim | FAKE | advanced | FAB+MISS+EMO | Schipper 2025 |
| 15 | fake_fact_checker_authority | FAKE | advanced | IMP+FAB+ANON | Tsipursky 2024 |
| 16 | coordinated_outrage_campaign | FAKE | proficient | POL+EMO+RECF | Ong & Cabañes 2018 |
| 17 | credible_verification_response | REAL | proficient | (none) | (REAL variety) |
| 18 | developing_situation_unverified | UNCERTAIN | novice | MISS | Schafer 2024 |

---

## Pre-pilot pack (HANDOFF.md P1.2)

The script `scripts/40_export_pilot_pack.py` builds a 30-card pack ready for SHS-student rater sessions. It produces:

- `printable_card_pack.html` — A4 print CSS, one card per page, 16pt body text
- `questionnaire.md` — 5 questions per card, ready to paste into Google Forms
- `answer_key.md` — gold verdict + DEPICT indicators + justifying phrase per card

Sampling is stratified (proportional verdict, proportional tier, balanced candidate, greedy tactic-novelty) and **seed-locked**, so the same pool + seed always produces the same 30 cards. Defense-reproducible.

---

## Citations

- **Roozenbeek, J., & van der Linden, S. (2019, 2020).** Bad News + Harmony Square — DEPICT taxonomy, "fictional examples throughout the game."
- **Costello et al. (2024) arxiv:2410.19202 (N=4,293 RCT).** — template + slots beats free-form.
- **Hainmueller, Hangartner, & Yamamoto (2015) PNAS 112(8).** — vignette-experiment "Smith vs. Jones" naming convention.
- **Garbe & Frischlich (2023) PLoS ONE.** — common-name labeling avoids real-world bias.
- **Arugay & Baquisal (2022) Pacific Affairs 95(3).** — Filipino dynastic / reformist / populist archetypes.
- **Schipper (2025) Data & Policy 7.** — Philippine 2025 disinformation playbook (deepfakes, recycled content).
- **Ong & Cabañes (2018).** — Filipino political trolling architecture.
- **Tsipursky (2024).** — fake fact-checker accounts.
- **Schafer et al. (2024) arxiv:2407.16051.** — ElectionRumors2022 dataset.
- **Yermilov et al. (2023) EACL.** — pseudonymization standard.
- **Pilán et al. (2022) Computational Linguistics 48(4).** — TAB anonymization benchmark.
- **Santos & del Rosario (2014).** — Philippine surname frequency.
- **Modirrousta-Galian & Higham (2023).** — credible-card mandate, per-tier calibration.
- **Powers (2011).** — precision/recall/F1 metric basis.
- **BATB_CompiledThesisPaper §1.3 p.12.** — "Rule-Constrained Content Generation" definition.

---

## License & academic use

This is undergraduate thesis software. Lead developer: Robert Gerald Senasin (FEU IT). All candidates, events, organizations, and narratives in the game are fictional and not intended to represent real individuals, political parties, or institutions (per thesis §1.5 Limitation #2).

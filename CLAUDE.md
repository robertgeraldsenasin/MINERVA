# M.I.N.E.R.V.A. — Project Briefing for Claude Code

> **You are working on the M.I.N.E.R.V.A. thesis project.** Read this file completely before making any changes. It tells you what the project is, how it's structured, what conventions to follow, and where the thinking has already been done.

---

## What this project is

M.I.N.E.R.V.A. is an Android Unity educational game that teaches Senior High School (SHS) Filipino voters to recognize misinformation about political candidates. It's a **psychological inoculation game** in the tradition of Cambridge's *Bad News* and *Harmony Square* — players see weakened doses of disinformation tactics so they build "mental antibodies" against them in the real world.

The thesis is at FEU IT (Far Eastern University, Institute of Technology). The defense is approaching. The user (Robert Gerald Senasin) is the lead developer.

**Repo:** `https://github.com/robertgeraldsenasin/MINERVA.git`
**Branch:** `upgrade/minerva-election-theme`
**Local clone path:** `D:\MINERVA\MINERVA\` (Windows, nested directory)

## What the user needs from you

**You do the work, I (the chat-side Claude) do the thinking.** When the user asks for something, they may have already discussed the approach with me in a parallel conversation. Your job is to execute the changes inside their actual repo, run the tests, and commit.

**Tight-loop conventions:**
1. **Read before writing.** Always view the relevant file(s) before editing.
2. **Run pytest after every change** (`python -m pytest tests/ -q`). 38/38 must pass before any commit.
3. **Run the full pipeline as a smoke test** when you change anything in scripts 21/23/24/26/28/30/31/32 (instructions below).
4. **Commit in small logical units** with descriptive messages. The user reviews commits before they push.
5. **Never reduce test coverage** without explicit user approval.
6. **Never edit `candidate_config.py` without confirmation** — that file is the user's editable config.

---

## Architecture (current state: v2.6-final-consolidated)

The pipeline is **template-based** (not GPT-2-augmented). The thesis paper §1.3 p.12 defines this approach as **"Rule-Constrained Content Generation"** — the architecture is the spec.

### The detection stack (already trained, weights frozen)

These models were trained on the JCBlaise Tagalog news corpus and are **not** what you should retrain casually:

- **RoBERTa-Tagalog** — 95.6% F1, fake/real binary classifier
- **DistilBERT-multilingual** — 91.0% F1, faster ensemble member
- **DE-GNN** — 95.8% F1, dual-channel embedding graph network
- **Random Forest + QLattice** — symbolic explainability layer

These produce fake-likelihood scores (`p_fake_robertaMINERVA`, etc.) that get ensembled into a final verdict. The Unity game doesn't see them directly — it shows the *cards* that come out of the pipeline.

### The card-generation pipeline (active development)

Scripts numbered 21+ produce the deck of teaching cards the game uses. Current state:

```
30 (templates)
  → 31 (universal pseudonymize, skip-template-cards)
  → 21 (balance verdict/candidate)
  → 23 (election-theme filter)
  → 24 (curate, enforce 7-day×8-card quotas)
  → 28 (draw per-user deck)
  → 26 (faithfulness audit)
  → 32 (detector validation NEW)
```

**17 templates × 18 per-template × 3 archetypes → 900 raw → 668-card pool.**

### Key concepts

- **DEPICT taxonomy** (Roozenbeek 2019): 12 disinformation indicators — EMO, URG, ANON, MISS, FAB, POL, CONS, DISC, IMP, REV, ENDO, RECF. Coverage: **12/12** in current pool.
- **Three archetypes** (Arugay 2022 Pacific Affairs): DYNASTIC, REFORMIST, POPULIST. Each candidate maps to one.
- **Three tiers** (Modirrousta-Galian 2023): novice / proficient / advanced explanation depth.
- **SIFT moves** (Caulfield's framework): STOP, INVESTIGATE, FIND, TRACE — the player's verdict response options.

---

## Editable config — `scripts/candidate_config.py`

**This is the most important file in the project.** It contains the three game candidates. The user can edit it; the entire pipeline picks up the change.

```python
CANDIDATES_CONFIG = [
    {"code": "C-A", "archetype": "DYNASTIC", "title": "Sen.",
     "first_name": "Ramon", "nickname": "Mon", "last_name": "Cruz",
     "region": "Northern Luzon", "aliases": ["Ramon", "Mon", "Cruz"]},
    {"code": "C-B", "archetype": "REFORMIST", "title": "Vice-Mayor",
     "first_name": "Liza", "nickname": "", "last_name": "Reyes",
     "region": "Central Visayas", "aliases": ["Liza", "Reyes"]},
    {"code": "C-C", "archetype": "POPULIST", "title": "Rep.",
     "first_name": "Joel", "nickname": "Joel", "last_name": "Garcia",
     "region": "Mindanao", "aliases": ["Joel", "Garcia"]},
]
```

### Constraints when editing this file (enforced at import)

1. Exactly 3 candidates
2. Archetypes must be exactly `{DYNASTIC, REFORMIST, POPULIST}`
3. Codes must match `^C-[A-Z0-9\-]{1,8}$`
4. Names must avoid surnames tied to current PH political families per thesis §1.5 Limitation #2 (the names below are common Filipino surnames per Santos & del Rosario 2014 PSA naming-frequency analysis — Cruz, Reyes, Garcia, Mendoza, Santos, Flores, Gonzales, Ramos, Bautista, Villanueva)

### What to NEVER do

- Do not hardcode `C-A`, `C-RM`, `Cruz`, `Marquez` etc. anywhere — always import from `candidate_config`
- Do not break the archetype-to-candidate binding (the disinformation tactics are bound to archetypes via `Arugay & Baquisal 2022`)
- Do not change the schema field names (`code`, `archetype`, `title`, `first_name`, `nickname`, `last_name`, `region`, `aliases`)

---

## The 18 templates (all in `scripts/30_template_scenario_generator.py`)

| # | Tactic | Verdict | Tier | Indicators | Source |
|---|---|---|---|---|---|
| 1 | historical_revisionism | FAKE | advanced | MISS+REV | Schipper 2025 |
| 2 | historical_revisionism_truth | REAL | novice | (none) | (REAL counterpart) |
| 3 | red_tagging | FAKE | proficient | EMO+FAB+ANON | Arugay 2022 |
| 4 | fake_celebrity_endorsement | FAKE | novice | EMO+ENDO+FAB | Ong & Cabañes 2018 |
| 5 | urgency_sharing | FAKE | novice | URG+MISS+EMO | Roozenbeek 2019 |
| 6 | fake_survey | FAKE | proficient | ENDO+FAB+MISS | Roozenbeek 2019 |
| 7 | credible_policy_announcement | REAL | novice | (none) | Modirrousta-Galian 2023 |
| 8 | ambiguous_allegation | UNCERTAIN | advanced | ANON+MISS | Schafer 2024 |
| 9 | conspiracy_theory | FAKE | advanced | CONS+ANON+MISS | DEPICT |
| 10 | polarization_us_vs_them | FAKE | proficient | POL+EMO+MISS | Roozenbeek 2020 |
| 11 | discrediting_personal_attack | FAKE | novice | DISC+EMO+ANON | Roozenbeek 2019 |
| 12 | fake_account_impersonation | FAKE | advanced | IMP+FAB+ANON | Ong & Cabañes 2018 |
| 13 | recycled_old_content | FAKE | novice | RECF+MISS+EMO | Schipper 2025 |
| 14 | deepfake_video_claim | FAKE | advanced | FAB+MISS+EMO | Schipper 2025 |
| 15 | fake_fact_checker_authority | FAKE | advanced | IMP+FAB+ANON | Tsipursky 2024 |
| 16 | coordinated_outrage_campaign | FAKE | proficient | POL+EMO+RECF | Ong & Cabañes 2018 |
| 17 | credible_verification_response | REAL | proficient | (none) | (REAL variety) |
| 18 | developing_situation_unverified | UNCERTAIN | novice | MISS | Schafer 2024 |

**Verified by `python -c "from scripts.thirty_template_scenario_generator import TEMPLATES; print(len(TEMPLATES))"` (or run `tests/test_template_count.py`).** If you add or remove a template, update this table.

If the user asks for **another tactic**, they want a new entry in `TEMPLATES`. Match the existing format. Each tactic needs:
- `tactic`: snake_case identifier
- `verdict`: `REAL` / `FAKE` / `UNCERTAIN`
- `archetypes`: list of which archetypes the tactic applies to
- `fired_indicators`: which DEPICT codes the explanation should fire
- `tier`: novice / proficient / advanced
- `templates`: list of 3+ template strings with `{slot}` placeholders

The slot fillers are at `fill_slots()` near line 446. Common slots: `candidate_full`, `candidate_short`, `candidate_first`, `prefix`, `place`, `platform`, `date`, `audience`, `generic_*`.

---

## How to run the pipeline (smoke test)

After any change to scripts 21/23/24/26/28/30/31/32 or any of the `minerva_*.py` modules, run this:

```powershell
# From D:\MINERVA\MINERVA\
$env:OUT = "tmp\smoke"
mkdir $env:OUT\generated -ErrorAction SilentlyContinue
mkdir $env:OUT\generated\decks -ErrorAction SilentlyContinue
mkdir $env:OUT\reports -ErrorAction SilentlyContinue

python scripts\30_template_scenario_generator.py --out_file $env:OUT\generated\template_cards.json --n_per_template 18 --report_out $env:OUT\reports\template_gen.json
python scripts\31_universal_pseudonymize.py --in_file $env:OUT\generated\template_cards.json --out_file $env:OUT\generated\cards_pseudo.json --report_out $env:OUT\reports\pseudo.json
python scripts\21_balance_unity_cards.py --in_file $env:OUT\generated\cards_pseudo.json --out_file $env:OUT\generated\balanced.json --target_total 700 --report_out $env:OUT\reports\balance.json
python scripts\23_enforce_election_theme.py --in_file $env:OUT\generated\balanced.json --out_file $env:OUT\generated\themed.json --report_out $env:OUT\reports\theme.json --rejection_log $env:OUT\reports\theme_rej.jsonl
python scripts\24_curate_teaching_cards.py --in_file $env:OUT\generated\themed.json --out_file $env:OUT\generated\pool.json --reject_out $env:OUT\generated\pool_rej.json --report_out $env:OUT\reports\pool.json --target_pool_size 700 --days 7 --cards_per_day 8 --min_credible_per_day 3 --seed 1729
python scripts\28_draw_user_deck.py --pool_file $env:OUT\generated\pool.json --out_dir $env:OUT\generated\decks --user_ids "alice,bob,charlie,diana,erika,fiona,greg,hana" --report_out $env:OUT\reports\draw.json
python scripts\26_faithfulness_audit.py --in_file $env:OUT\generated\pool.json --report_out $env:OUT\reports\faith.json
python scripts\32_validate_detectors_on_templates.py --pool_file $env:OUT\generated\pool.json --report_out $env:OUT\reports\det.json --markdown_out $env:OUT\reports\det.md
```

**Expected metrics (regression targets):**
- Pool size: 668 (±15)
- Faithfulness audit: 100% (no regressions allowed)
- Pairwise overlap mean: < 13% (consolidated achieved 11.48%)
- Detector validation accuracy: 100% (with `uncertain_band=0.05`)
- 12/12 indicator coverage

**If any metric regresses, do not commit.** Investigate first — usually it means a template's text is missing the indicator-mention vocabulary the audit lexicon checks for.

---

## Conventions

### File layout

```
D:\MINERVA\MINERVA\
├── scripts\           # All Python pipeline scripts
│   ├── candidate_config.py        # EDITABLE — three candidates
│   ├── 30_template_scenario_generator.py  # the 17 templates
│   ├── 31_universal_pseudonymize.py
│   ├── 32_validate_detectors_on_templates.py
│   ├── 26_faithfulness_audit.py
│   ├── minerva_candidates.py    # rebuilds REGISTRY from config
│   ├── minerva_filters.py
│   ├── minerva_schemas.py
│   ├── minerva_indicators.py
│   └── minerva_response_bank.py
├── tests\             # pytest test suite
│   └── test_filters.py
├── docs\              # design docs
│   ├── V2.6_CHANGES.md           # READ THIS FIRST
│   ├── MASTER_CODEBOOK.md
│   └── MINERVA_v2.6_Audit.html
├── notebooks\         # Colab pipeline runner
│   └── MINERVA_Run_Colab_v2.6.ipynb
├── generated\         # pipeline outputs (gitignored)
└── reports\           # pipeline reports (gitignored)
```

### Style

- Python 3.10+ syntax (use `dict | None`, not `Optional[dict]`)
- Docstrings cite the research that justifies any non-obvious decision
- Type hints on every public function
- Logging via `logger = logging.getLogger(__name__)`, not print statements
- Subprocess invocation: prefer `python scripts/...` (uses the project's venv)

### Git workflow

- Branch: `upgrade/minerva-election-theme` (do not switch unless asked)
- Commit messages start with `feat:` / `fix:` / `docs:` / `test:` / `refactor:`
- Reference the issue or feature in the commit body
- Never commit `generated/` or `reports/` outputs (they're already gitignored)
- Show the user the diff before pushing if the change is non-trivial

---

## What's been done (chronologically)

If the user references a previous version, this is what each one means:

- **v2.0** — initial refactor with 12-cue DEPICT taxonomy, 3-candidate registry, faithfulness audit
- **v2.1-v2.4** — bug fixes, post-audit hardening, dynamic-content 500-card pool, three-candidate enforcement
- **v2.5** — text post-processing patches (truncation 93.5%→2%, name-jam 50%→28%)
- **v2.6** — template-based generation replaces GPT-2; pool 472, 8/12 indicators, 100% faithfulness
- **v2.6-final** — editable common-name candidate config + Pattern E single-name catch
- **v2.6-final-consolidated (CURRENT)** — 17 templates × 18 × 3 archetypes = 668 pool, 12/12 indicators, 11.48% pairwise overlap, 100% detector validation, all 7 issues closed

The user may say "v2.7" but they don't actually want a v2.7 — they want everything finished as v2.6.

---

## Citations the project relies on

When justifying a code decision, cite from this list. The full list is in `docs/V2.6_CHANGES.md`.

- **Roozenbeek & van der Linden (2019, 2020)** — DEPICT taxonomy, scripted scenarios, "fictional examples throughout the game"
- **Costello et al. (2024) arxiv:2410.19202, N=4,293 RCT** — template + slots beats free-form
- **Hainmueller, Hangartner, & Yamamoto (2015) PNAS 112(8)** — vignette-experiment "Smith vs. Jones" naming convention
- **Garbe & Frischlich (2023) PLoS ONE** — common-name labeling avoids real-world bias
- **Arugay & Baquisal (2022) Pacific Affairs 95(3)** — Filipino dynastic/reformist/populist archetypes
- **Schipper (2025) Data & Policy 7** — Philippine 2025 disinformation playbook (deepfakes, recycled content)
- **Ong & Cabañes (2018)** — Filipino political trolling architecture
- **Tsipursky (2024)** — fake fact-checker accounts as 2024+ tactic
- **Schafer et al. (2024) arxiv:2407.16051** — ElectionRumors2022 dataset
- **Yermilov et al. (2023) EACL** — pseudonymization standard
- **Pilán et al. (2022) Computational Linguistics 48(4)** — TAB anonymization benchmark
- **Santos & del Rosario (2014)** — Philippine surname frequency data
- **Modirrousta-Galian & Higham (2023)** — credible-card mandate, per-tier calibration
- **Powers (2011)** — precision/recall/F1 metric basis
- **BATB_CompiledThesisPaper §1.3 p.12** — "Rule-Constrained Content Generation" definition

---

## What you should ask the user before doing

If any of these come up, **ask before acting**:

- "Should I retrain RoBERTa / DistilBERT / DE-GNN?" (almost always no — these are frozen)
- "Should I change `candidate_config.py`?" (almost always no — that's the user's file)
- "Should I add new dependencies?" (probably no — the project keeps `requirements.txt` minimal)
- "Should I delete X?" (always confirm, even if X looks redundant)
- "Should I push to a new branch?" (no — stay on `upgrade/minerva-election-theme` unless told)

If any of these come up, **just do it without asking**:

- Adding a new template (just match the format of existing entries in `TEMPLATES`)
- Expanding the audit lexicon for a new indicator marker
- Adding a test that captures a regression you noticed
- Updating `V2.6_CHANGES.md` with a one-paragraph note about a fix
- Running the smoke-test pipeline after a change

---

## Where to find more context

- `docs/V2.6_CHANGES.md` — full history of v2.6 → consolidated, with rationale and metric tables
- `docs/MASTER_CODEBOOK.md` — coding conventions and explanation-bank schema
- `docs/MINERVA_v2.6_Audit.html` — visual audit (open in browser, has 10 sections)
- `HANDOFF.md` (in this bundle's root) — open work items the chat-side Claude has flagged for you
- `README.md` (in this bundle's root) — install + invocation instructions for the user

When in doubt, ask the user. They've been thinking about this for months and will catch a wrong direction faster than you can fix it.

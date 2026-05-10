# M.I.N.E.R.V.A. — Final Branch (`v2.9.1`)

**Misinformation Identification through Neuro-Symbolic Evaluation, Reasoning, and Verifiable Analytics**

A Tagalog-language educational content pipeline that combines hybrid credibility detection (DE-GNN + Random Forest + RoBERTa/DistilBERT ensemble) with QLattice symbolic regression and a CTRL-style conditioned GPT-2 to generate, score, and curate a teaching pool of Filipino electoral-misinformation cards for senior high school media literacy use.

[![tests](https://img.shields.io/badge/tests-231%20passed-brightgreen)]() [![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)]() [![colab](https://img.shields.io/badge/runtime-Colab%20T4-orange)]() [![release](https://img.shields.io/badge/release-v2.9.1-blue)]()

---

## What this repository delivers (Thesis 2 scope)

This is the **content-and-scoring pipeline plus the curated card pool** — the upstream feedstock for the Unity Android game. The Unity build itself, the 50-respondent SHS pilot study, and the 5-evaluator ISO 25010 review are scoped to **Thesis 3**.

What runs end-to-end on a free Colab T4 in ~70-80 minutes:

1. Train RoBERTa-Tagalog and DistilBERT-multilingual fake-news detectors on the JCBlaise dataset (5 random seeds, val-then-test selection).
2. Train a Dual-Embedding Graph Neural Network (DE-GNN) and a Random Forest classifier on transformer-derived features.
3. Fit a QLattice symbolic-regression equation over detector confidence + DE-GNN signal for interpretable scoring.
4. Fine-tune `jcblaise/gpt2-tagalog` with 18 Keskar-style control tokens conditioning on (label × graph confidence × QLattice confidence × ensemble confidence × tier).
5. Generate, score, pseudonymize, and curate a 668-card teaching pool with 12 misinformation indicators tied to SIFT moves.
6. Pseudonymize all real Filipino political figures (people *and* places — v2.9.0) into Candidate A/B/C and City W/X/Y/Z placeholders.
7. Audit faithfulness (≥98% pass), strict allowlist (≥99% pass), and indicator coverage end-to-end.
8. Draw 8 user-specific 56-card decks ready to feed a downstream Unity game build.

---

## Repository layout

```
.
├── scripts/                    # 48 pipeline scripts numbered 01-40 + helpers
│   ├── 01_download_dataset.py
│   ├── 04_train_robertaMINERVA.py
│   ├── 05_train_distilbertMINERVA.py
│   ├── 08_train_qlattice.py
│   ├── 09_train_degnn.py
│   ├── 10b_prepare_gpt2_neurosymbolic.py    # UPDATED v2.9.0
│   ├── 11b_train_gpt2_neurosymbolic.py
│   ├── 12b_generate_gpt2_neurosymbolic.py
│   ├── 17_run_5seeds_detectors.py
│   ├── 29_merge_gpt2_into_pool.py           # UPDATED v2.9.0
│   ├── 33_strict_name_allowlist.py
│   ├── 35_pseudonymize_places.py            # NEW v2.9.0
│   ├── 37_holdout_detector_eval.py          # NEW v2.9.0
│   └── 40_export_pilot_pack.py
├── tests/                      # 231 unit tests, runs in <3s
│   ├── test_pseudonymize_places.py          # NEW v2.9.0
│   ├── test_response_bank.py                # NEW v2.9.0
│   ├── test_holdout_eval.py                 # NEW v2.9.0
│   ├── test_degnn_graph.py                  # UPDATED v2.9.1 (clean torch skip)
│   └── ... (13 more)
├── templates/                  # Hand-curated content artifacts
│   ├── candidate_profiles_three_candidates.json
│   ├── jcblaise_real_names_blocklist.txt
│   ├── places_blocklist.txt                 # NEW v2.9.0 (171 entries)
│   ├── response_bank_v2.json                # NEW v2.9.0 (225 phrases)
│   ├── holdout_gpt2_labeled.csv             # NEW v2.9.0 (50-card seed)
│   └── holdout_gpt2_labeled.README.md       # NEW v2.9.0 (labeling protocol)
├── notebooks/
│   ├── MINERVA_Run_Colab_v2.9.0.ipynb       # ← canonical (use this one)
│   └── legacy/
│       └── MINERVA_Run_Colab_v2.8.7.ipynb   # superseded; reference only
├── docs/
│   ├── V2.9.0_RELEASE_NOTES.md
│   ├── V2.9.0_AUDIT_RESPONSE.md             # point-by-point response to v2.8.7 audit
│   ├── V2.9.1_HOTFIX_NOTES.md
│   └── V2.9.1_STEP_BY_STEP_FIX.md           # apply + verify guide
├── requirements.txt                         # full pipeline (~3 GB; for Colab)
├── dev-requirements.txt                     # NEW v2.9.1 (lightweight ~150 MB; for local pytest)
├── CHANGELOG.md                             # NEW v2.9.1 (full version history)
├── .gitignore                               # NEW v2.9.1
└── README.md                                # this file
```

Empty `data/`, `logs/`, `models/`, `generated/`, `reports/` directories are kept (with `.gitkeep` markers) because the pipeline scripts write into them at runtime.

---

## Quickstart

### Local (test suite only — recommended for development)

```powershell
# Windows + Python 3.11 or 3.12
git clone -b final_ver_branch https://github.com/robertgeraldsenasin/MINERVA.git MINERVA_FINAL
cd MINERVA_FINAL
C:\Python312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r dev-requirements.txt
python -m pytest tests/ -q
# Expected: 231 passed, 1 skipped in ~3s
```

The 1 skipped test is `test_degnn_graph.py` skipping cleanly because torch isn't in the dev dependencies. **This is correct behavior.**

### Pipeline run (Colab T4)

1. Open Colab → File → Open notebook → GitHub tab
2. Repo: `robertgeraldsenasin/MINERVA` · Branch: `final_ver_branch`
3. Open `notebooks/MINERVA_Run_Colab_v2.9.0.ipynb`
4. Runtime → Change runtime type → **T4 GPU**
5. Runtime → Restart runtime → Run all
6. Wall time: ~70-80 minutes

After the run, verify the four headline numbers per `docs/V2.9.1_STEP_BY_STEP_FIX.md::Step 7`.

---

## Architectural truths (do not relitigate)

These are decisions finalized during Thesis 1 and BATB §1.5. **Don't change them in this branch.**

1. **Codes-only candidates.** The pool only references `Candidate A`, `Candidate B`, `Candidate C`. No real political figures.
2. **Control-token GPT-2 (Keskar CTRL-style)**, not LoRA. 18 special tokens across 5 fields: label × graph × QLattice × ensemble × tier confidence bins.
3. **Strict allowlist enforcer (script 33) is the safety net.** Multi-pass; rejects any card mentioning unknown names. Runs after pseudonymize.
4. **JCBlaise dataset**: `https://huggingface.co/datasets/jcblaise/fake_news_filipino/resolve/main/fakenews.zip` (3,206 records).
5. **DE-GNN → Random Forest sequential pipeline** per BATB §3.5.2 (was parallel in v2.7).
6. **Pool target: 668 cards** (108 REAL / 500 FAKE / 60 UNCERTAIN).

---

## Version history

See [`CHANGELOG.md`](CHANGELOG.md) for full version history. Recent highlights:

| Version | Date | Headline |
|---|---|---|
| **v2.9.1** | 2026-05-08 | Hotfix: dev-requirements.txt + clean torch skip in tests + Python compatibility docs |
| **v2.9.0** | 2026-05-08 | **Audit-driven refinement: place-name pseudo + response bank + version assertion + holdout eval** |
| v2.8.7 | 2026-05-05 | GPT-2 → pool merge + persistent generation loop + sentence recovery |
| v2.8.6 | 2026-04-30 | Percentile binning + GPT2 epochs 3→8 |
| v2.8.5 | 2026-04-28 | Script 11b dataset bypass |
| v2.8.4 | 2026-04-25 | feyn pin >=3.4,<4.0 + sklearn LogReg fallback |
| v2.8.3 | 2026-04-22 | transformers 4.46+ API |
| v2.8.2 | 2026-04-20 | Dataset download fix (JCBlaise direct ZIP) |

---

## What this branch does NOT include

So the panel knows where the boundaries are:

- **Unity Android build.** Thesis 3 deliverable.
- **Live SHS pilot data (50 respondents).** Thesis 3.
- **ISO 25010 evaluator review (5 evaluators).** Thesis 3.
- **Decoder-time rule-constrained generation.** Currently uses post-hoc QLattice filtering (script 13), not constrained decoding. The paper text should describe it as such — see `docs/V2.9.0_AUDIT_RESPONSE.md::P2 #1`.

---

## Documentation

| Document | When to read it |
|---|---|
| `CHANGELOG.md` | Looking up what changed between any two versions |
| `docs/V2.9.0_RELEASE_NOTES.md` | What v2.9.0 delivered, in depth |
| `docs/V2.9.0_AUDIT_RESPONSE.md` | Point-by-point response to the v2.8.7 audit findings (panel-facing) |
| `docs/V2.9.1_HOTFIX_NOTES.md` | What v2.9.1 added on top of v2.9.0 |
| `docs/V2.9.1_STEP_BY_STEP_FIX.md` | Apply + verify guide for the v2.9.x release |

---

## Citation

If used in academic work, cite:
- **JCBlaise dataset**: Cruz, Tan, & Cheng (2020). *Localization of Fake News Detection*. LREC.
- **CTRL conditioning**: Keskar et al. (2019). *CTRL: A Conditional Transformer Language Model for Controllable Generation*.
- **QLattice / feyn**: Brolos et al. (2021). *An Approach to Symbolic Regression Using Feyn*.
- **GraphSAGE**: Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs*.

# M.I.N.E.R.V.A. — Refinement Branch (`v2.9.0`)

**Misinformation Identification through Neuro-Symbolic Evaluation, Reasoning, and Verifiable Analytics**

A Tagalog-language educational content pipeline that combines hybrid credibility detection (DE-GNN + Random Forest + RoBERTa/DistilBERT ensemble) with QLattice symbolic regression and a CTRL-style conditioned GPT-2 to generate, score, and curate a teaching pool of Filipino electoral-misinformation cards for senior high school media literacy use.

---

## What this repository delivers (Thesis 2 scope)

This is the **content-and-scoring pipeline plus the curated card pool** — the upstream feedstock for the Unity Android game. The Unity build itself, the 50-respondent SHS pilot study, and the 5-evaluator ISO 25010 review are scoped to **Thesis 3**.

What runs end-to-end on a free Colab T4 in ~75 minutes:

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
MINERVA_REFINED/
├── scripts/                    # 48 pipeline scripts numbered 01-40 + helpers
│   ├── 01_download_dataset.py
│   ├── ...
│   ├── 29_merge_gpt2_into_pool.py     # NEW v2.8.7, refined v2.9.0
│   ├── 35_pseudonymize_places.py      # NEW v2.9.0
│   ├── 37_holdout_detector_eval.py    # NEW v2.9.0
│   └── 40_export_pilot_pack.py
├── templates/                  # Hand-curated content artifacts
│   ├── candidate_profiles_three_candidates.json
│   ├── jcblaise_real_names_blocklist.txt
│   ├── places_blocklist.txt           # NEW v2.9.0 — 171 PH geographic entities
│   ├── response_bank_v2.json          # NEW v2.9.0 — 225 phrases, 72% diverse
│   └── holdout_gpt2_labeled.csv       # NEW v2.9.0 — 50-card hand-label seed
├── tests/                      # 231 unit tests, runs in <3s
│   ├── test_pseudonymize_places.py    # NEW v2.9.0
│   ├── test_response_bank.py          # NEW v2.9.0
│   └── test_holdout_eval.py           # NEW v2.9.0
├── notebooks/
│   └── MINERVA_Run_Colab_v2.9.0.ipynb # 71 cells, end-to-end pipeline
├── docs/
│   ├── V2.9.0_RELEASE_NOTES.md        # what changed and why
│   ├── V2.9.0_AUDIT_RESPONSE.md       # point-by-point response to v2.8.7 audit
│   └── V2.9.0_STEP_BY_STEP_FIX.md     # apply + verify guide
└── requirements.txt            # transformers ≥4.46, fsspec ≤2024.6.1, feyn ≥3.4 <4.0
```

---

## Quickstart (local, for testing only)

The full pipeline needs a GPU; locally only run the test suite:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/ -q
# Expected: 231 passed in ~3 seconds
```

For the actual pipeline run, use `notebooks/MINERVA_Run_Colab_v2.9.0.ipynb` on Colab T4.

---

## Architectural truths (do not relitigate)

These are decisions that were finalized during Thesis 1 and BATB §1.5. Don't change them in this branch:

1. **Codes-only candidates.** The pool only references `Candidate A`, `Candidate B`, `Candidate C`. No real political figures.
2. **Control-token GPT-2 (Keskar CTRL-style)**, not LoRA. 18 special tokens across 5 fields: 5 labels × 4 graph bins × 4 QLattice bins × 4 ensemble bins × 4 tier bins.
3. **Strict allowlist enforcer (script 33) is the safety net.** Multi-pass; rejects any card mentioning unknown names. Runs after pseudonymize.
4. **JCBlaise dataset**: `https://huggingface.co/datasets/jcblaise/fake_news_filipino/resolve/main/fakenews.zip` (3,206 records).
5. **Pipeline order**: detectors → DE-GNN → Random Forest → QLattice → GPT-2 fine-tune → generation → score → pseudonymize → balance → theme → curate → strict allowlist.
6. **Pool target: 668 cards** (108 REAL / 500 FAKE / 60 UNCERTAIN).

---

## Version history

| Version | Date | Headline |
|---|---|---|
| v2.6.final | 2026-04 | Initial neuro-symbolic stack |
| v2.8.2 | 2026-04 | Dataset download fix (JCBlaise direct ZIP bypass) |
| v2.8.3 | 2026-04 | transformers 4.46+ API + IPython traceback workaround |
| v2.8.4 | 2026-04 | feyn 3.x pin + sklearn LogReg fallback for QLattice |
| v2.8.5 | 2026-04 | Script 11b dataset bypass (`Dataset.from_dict`) |
| v2.8.6 | 2026-04 | Percentile binning + GPT2 epochs 3→8 + differentiated seeds |
| v2.8.7 | 2026-05 | Script 29 GPT-2 → pool merge fix + sentence recovery + Excel-style code remap |
| **v2.9.0** | 2026-05 | **Audit-driven refinement: place-name pseudo + response bank + version assertion + holdout eval** |

See `docs/V2.9.0_RELEASE_NOTES.md` for details on the latest release.

---

## What v2.9.0 explicitly does NOT include

So the panel knows where the boundaries are:

- **Unity Android build.** Thesis 3.
- **Live SHS pilot data.** Thesis 3.
- **ISO 25010 evaluator review.** Thesis 3.
- **Decoder-time rule-constrained generation.** The architecture's "rule-constrained generation" claim currently uses post-hoc QLattice filtering, not constrained decoding. The paper text should describe it accordingly. See `docs/V2.9.0_AUDIT_RESPONSE.md::P2 #1`.

---

## Citation

If used in academic work, cite the JCBlaise dataset (Cruz, Tan, & Cheng 2020, LREC), the CTRL conditioning approach (Keskar et al. 2019), and the QLattice symbolic regression library (Brolos et al. 2021).

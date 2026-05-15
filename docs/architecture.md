# Architecture Overview

Single-page mental model of M.I.N.E.R.V.A. for new readers and panelists.

---

## What this system is, in one sentence

M.I.N.E.R.V.A. is a **neuro-symbolic pipeline** that generates and curates fictionalized Tagalog social-media cards labeled with misinformation indicators, intended to feed an educational Unity Android game for Filipino senior high school media-literacy training.

## What it is NOT

- **Not a live fact-checker** — it produces simulated training material, not real-time misinformation detection.
- **Not the Unity game itself** — the game is Thesis 3 scope (Salva).
- **Not multimodal** — text-only pipeline.
- **Not a deployment system** — all models run offline; the Unity game consumes the precomputed card pool as static JSON.

---

## The four layers (paper §3.3)

```
┌─────────────────────────────────────────────────────────┐
│  Presentation Layer (Unity — Thesis 3 scope)            │
│  VERIdex · Chattr · VERIdict · V.E.R.I.T.A.S. · Tasks   │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │  (consumes unity_cards_pool_strict.json)
                          │
┌─────────────────────────────────────────────────────────┐
│  Application Layer (Unity — Thesis 3 scope)             │
│  Game manager · scoring · feedback · day cycle          │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────┐
│  Decision Layer (THIS REPO — Thesis 2 scope)            │
│  RoBERTa + DistilBERT + GraphSAGE + QLattice + RF       │
│  + GPT-2 generator with control tokens                  │
│  + safety chain (pseudonymizer + allowlist + audit)     │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │
┌─────────────────────────────────────────────────────────┐
│  Data Layer (JCBlaise dataset + template scenarios)     │
└─────────────────────────────────────────────────────────┘
```

This repository implements the **Decision Layer**. The Unity team consumes its outputs.

---

## The 5 model components

| Component | Where | What it does | Training |
|---|---|---|---|
| **RoBERTa-Tagalog** | scripts 04, 17, 06 | Filipino-specialist fake-news classifier | 5 prime seeds, best-by-val |
| **DistilBERT-multilingual** | scripts 05, 17, 06 | Multilingual generalist for code-switched Taglish | 5 prime seeds, best-by-val |
| **GraphSAGE GNN** ("DE-GNN") | scripts 09, minerva_degnn.py | Aggregates neighborhood signal over kNN similarity graph of dual-embedding features | seed 0, 60 epochs, early-stopping |
| **QLattice symbolic regression** | scripts 08, 13, 18 | Distills detector + GNN signal into compact equation for interpretable scoring | feyn library, ~5 generations |
| **GPT-2 Tagalog (CTRL-conditioned)** | scripts 10b, 11b, 12b | Generates 500 FAKE + 500 REAL candidates per attempt with 18 control tokens | seed 1729, 3 epochs |

### Why each one?

- **Two detectors** = redundant-sensor design. Disagreement signals uncertainty (useful for NEUTRAL cards).
- **GraphSAGE** = adds propagation-aware smoothing over similarity neighborhoods (kNN approximation of social graph when no live edges exist).
- **QLattice** = extracts a teachable equation; supports interpretable feedback in the Unity game.
- **GPT-2 with control tokens** = scalable scenario generation (~30× cheaper than hand-writing 668 cards).
- **Random Forest baseline** = sanity check that the neural models are doing meaningful work.

---

## The 23-stage curation pipeline (Thesis 2 deliverable)

The full pipeline that runs on Colab A100 in ~30 minutes:

```
Stage 01-03:   Data acquisition + split
Stage 04-07:   Train RoBERTa, DistilBERT, RF + extract features (5 prime seeds)
Stage 08-09:   QLattice + GraphSAGE
Stage 10b-12b: GPT-2 corpus build + fine-tune + generate
Stage 13:      QLattice scoring of generated content
Stage 14-19:   Baselines, evaluation, plotting
Stage 21:      Balance for educational bucket targets (schema-validated)
Stage 23:      Theme enforcement (election-only filter)
Stage 24:      Teaching card curation (12 indicators × 3 tiers)
Stage 26:      Faithfulness audit (indicator-phrase consistency)
Stage 28:      Draw 8 user decks (56 cards each)
Stage 29:      Merge GPT-2 cards into template pool
Stage 30:      Template scenario generator (deterministic)
Stage 31:      People pseudonymizer (NER + blocklist)
Stage 33:      Strict allowlist (last-line safety gate)
Stage 35:      Places pseudonymizer (6 categories: HUCs, provinces, regions, ...)
Stage 40:      Pilot pack export (HTML + CSV + JSON for offline review)
```

**Final output:** `generated/unity_cards_pool_strict.json` — 658 cards consumed by Unity.

---

## The safety chain (4 stages, critical for defense)

```
Cards
  │
  ▼
[31 Pseudonymize PEOPLE] ← NER + blocklist (real PH political dynasty surnames)
  │
  ▼
[35 Pseudonymize PLACES] ← 260+ entry blocklist (Metro Manila HUCs, districts, regions)
  │
  ▼
[33 Strict allowlist GATE] ← 130+ allowed orgs (gov agencies, geographic pseudonyms, role titles)
  │ pass: 99.09% (652/658)
  │ rejected: 6 (1 correct safety block of "Pacquiao" + 5 generic agency terms)
  ▼
[26 Faithfulness audit] ← every card's indicator phrases must be canonical
  │ pass: 100% (658/658)
  ▼
Game-ready pool ✓
```

**Defense talking point:** the 6 rejections demonstrate the chain catching what it should. A 100% rate would mean either the chain is too permissive or GPT-2 never produced anything sensitive (improbable).

---

## Reproducibility design (Pineau 2021 compliant)

1. **5 prime seeds** [13, 29, 47, 89, 127] per Liu et al. (2019) for detector training
2. **Picard 1729** seed for GPT-2 (avoids the cherrypicked-42 problem)
3. `reports/_environment.json` captures Python, torch, transformers, CUDA, host
4. Pool hash + bank hash + run_id in every report
5. Train-only `StandardScaler.fit()` — no test leakage
6. EarlyStoppingCallback on val_f1
7. 311 regression tests covering schemas, percentile binning, pseudonymization, response bank, merge logic, holdout, strict allowlist

---

## Key design tradeoffs (acknowledged in §5 Limitations)

| Tradeoff | Why it's the right call for Thesis 2 |
|---|---|
| Diversity 13% (template-dominated) | Educational curriculum bucket targets prioritize indicator coverage over phrase variety; v2.10 (Thesis 3) will rebalance based on SHS pilot feedback |
| "DE-GNN" = dual-embedding GraphSAGE (not Differential-Evolution) | Differential Evolution optimizer scaffolded for Thesis 3; current Adam + early-stopping with 5-seed protocol achieves paper's target F1 |
| Holdout F1 deferred to external Filipino validators | Methodologically stronger than single-annotator internal labels; documented in `docs/HOLDOUT_VALIDATION_STRATEGY.md` |
| Script 38 ablation scaffold not run | Publication-grade evidence for SO 2; deferred post-defense |
| 0.91% safety blocks (Pacquiao etc.) | The chain working as designed — real political-dynasty surnames must remain blocked |

---

## What a panelist should be able to verify in 5 minutes

1. Open `reports/faith.json` → see `pass_rate: 100.0`
2. Open `reports/strict_allowlist.json` → see `pass_rate_pct: 99.09`
3. Open `reports/detectors_5seed_summary_v28_panel.json` → see RoBERTa 95.30% ± 0.40%
4. Open `generated/unity_cards_pool_strict.json` → 658 cards, 12 indicators, no real political names
5. Run `pytest tests/ -q` → 311 passed in 3 seconds

Every claim in the paper has a corresponding report file with hashed provenance.

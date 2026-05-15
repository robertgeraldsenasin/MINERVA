# M.I.N.E.R.V.A. — Master Codebook

Centralized technical documentation. Reading this end-to-end gives you everything needed to understand, reproduce, validate, or extend the system without reading inline comments scattered across scripts.

**Reading order:** §1 → §2 → §10 for a 10-minute overview. Read §3-§9 only when working on the specific component.

---

## §1 — Repository structure

```
MINERVA/
├── scripts/                # Numbered pipeline (01-40) + helper modules
│   ├── 01-19_*.py          # Training pipeline (data → models)
│   ├── 21-40_*.py          # Curation pipeline (models → card pool)
│   ├── minerva_*.py        # 8 helper modules (config, schemas, etc.)
│   └── legacy/             # Superseded scripts kept for reference
├── tests/                  # 311 unit tests
├── notebooks/
│   ├── MINERVA_Run_Colab_v2.9.6.ipynb     # Canonical run notebook
│   └── legacy/             # Previous iterations
├── templates/              # Hand-curated content (response bank, blocklists)
├── docs/                   # Strategy + release docs (this file lives here)
├── data/, generated/, models/, reports/, logs/   # Runtime outputs (gitignored)
├── requirements.txt        # Full pipeline dependencies (~3 GB)
├── dev-requirements.txt    # Lightweight test-only (~150 MB)
├── CHANGELOG.md            # Version history v2.6 → v2.9.9
├── README.md               # Project overview
└── Makefile                # Convenience targets
```

**Active script count:** 47 (44 pipeline + 8 helpers, with 2 superseded scripts retained in `scripts/legacy/` for traceability).

---

## §2 — Pipeline overview (23 stages)

```
DATA → MODELS → GENERATION → CURATION → SAFETY CHAIN → EXPORT
```

### Phase A — Data acquisition (stages 01-03)
| Script | Purpose | Output |
|---|---|---|
| `01_download_dataset.py` | Pull JCBlaise Filipino fake-news dataset from HuggingFace | `data/raw/` |
| `02_prepare_dataset.py` | Normalize columns, sanitize text | `data/processed/` |
| `03_split_dataset.py` | Stratified 70/15/15 train/val/test split with fixed seed | `data/splits/` |

### Phase B — Detector training (stages 04-07, 14-17)
| Script | Purpose | Output |
|---|---|---|
| `17_run_5seeds_detectors.py` | Orchestrate detector training across 5 prime seeds [13, 29, 47, 89, 127] | per-seed model dirs |
| `04_train_robertaMINERVA.py` | RoBERTa-Tagalog fine-tune (called per seed) | `models/roberta/seed_*` |
| `05_train_distilbertMINERVA.py` | DistilBERT-multilingual fine-tune | `models/distilbert/seed_*` |
| `07_train_random_forest.py` | RF baseline on tabular features | `models/rf.joblib` |
| `14_train_baseline_tfidf_logreg.py` | TF-IDF + LogReg baseline | `models/tfidf_lr.joblib` |
| `15_evaluate_detectors.py` | Test-set evaluation, best-by-val export | `reports/detectors_*.json` |
| `export_best_detectors_by_val.py` | Helper: select best seed via validation | `models/*_finetuned/` |

### Phase C — Feature extraction + secondary models (stages 06, 08-09)
| Script | Purpose | Output |
|---|---|---|
| `06_extract_features.py` | CLS embeddings + PCA → tabular features | `data/features/` |
| `08_train_qlattice.py` | Symbolic regression (`feyn` library) → compact equation | `models/qlattice_equation.txt` |
| `09_train_degnn.py` | GraphSAGE on kNN similarity graph | `models/degnn.pt` |

### Phase D — Conditioned generation (stages 10b-13)
| Script | Purpose | Output |
|---|---|---|
| `10b_prepare_gpt2_neurosymbolic.py` | Build training corpus with 18 Keskar-style control tokens | `data/gpt2/corpus.txt` |
| `11b_train_gpt2_neurosymbolic.py` | Fine-tune `jcblaise/gpt2-tagalog` (seed 1729) | `models/gpt2_neuro/` |
| `12b_generate_gpt2_neurosymbolic.py` | Generate 500 FAKE + 500 REAL candidates | `generated/_a0_*.jsonl` |
| `13_score_generated_with_qlattice.py` | Score each candidate with QLattice equation | scored JSONL |

### Phase E — Curation pipeline (stages 21-30)
| Script | Purpose | Output |
|---|---|---|
| `29_merge_gpt2_into_pool.py` | Merge GPT-2 candidates + template scenarios | `generated/template_plus_gpt2_cards.json` |
| `30_template_scenario_generator.py` | Deterministic template-based card generation | `generated/template_cards.json` |
| `31_universal_pseudonymize.py` | NER + blocklist person-name pseudonymization | `generated/cards_pseudo.json` |
| `35_pseudonymize_places.py` | 260+ entry place-name blocklist applied | `generated/cards_pseudo_places.json` |
| `21_balance_unity_cards.py` | Bucket-fill across indicator × tier × verdict; schema validate | `generated/balanced.json` |
| `23_enforce_election_theme.py` | Reject off-theme content | `generated/themed.json` |
| `24_curate_teaching_cards.py` | Final teaching-card curation | `generated/unity_cards_pool.json` |
| `25_build_candidate_scenarios.py` | Build per-candidate scenario fixtures | `templates/scenarios/` |

### Phase F — Safety chain + audit (stages 26, 33)
| Script | Purpose | Output |
|---|---|---|
| `33_strict_name_allowlist.py` | Last-line safety gate; allowlist of 130+ orgs | `generated/unity_cards_pool_strict.json` |
| `26_faithfulness_audit.py` | Verify every card's indicator phrases are canonical | `reports/faith.json` |
| `34_extract_jcblaise_names.py` | Build training-data blocklist | `templates/jcblaise_real_names_blocklist.txt` |

### Phase G — Deck draw + export (stages 28, 40)
| Script | Purpose | Output |
|---|---|---|
| `28_draw_user_deck.py` | Draw 8 user-specific 56-card decks with overlap control | `generated/decks/` |
| `40_export_pilot_pack.py` | Export pilot pack (HTML + CSV + JSON) | `reports/pilot_pack/` |

### Phase H — Evaluation utilities (stages 18-19, 32, 37-38)
| Script | Purpose | Output |
|---|---|---|
| `18_verdict_explain.py` | Generate per-card explanation strings | (inline) |
| `19_plot_training_graphs.py` | Training-curve visualizations | `reports/figures/` |
| `32_validate_detectors_on_templates.py` | Sanity-check detectors vs templates | `reports/det.json` |
| `37_holdout_detector_eval.py` | Holdout eval (deferred to external validators) | optional |
| `38_ablation_no_conditioning.py` | Ablation scaffold for SO 2 publication evidence | optional |

---

## §3 — Dataset flow

**Source:** JCBlaise Filipino fake-news dataset (HuggingFace: `jcblaise/fake_news_filipino`).
**Size:** ~3,206 records.
**Schema:** `{text: str, label: int}` where label 0 = real, 1 = fake.

**Split:** Stratified 70/15/15 with fixed seed via `sklearn.model_selection.train_test_split`. Documented in `scripts/03_split_dataset.py`.

**Provenance:** Each split CSV ships with a hash logged in `_environment.json` so any run is verifiable end-to-end.

---

## §4 — Preprocessing pipeline

### Text normalization (`scripts/02_prepare_dataset.py`)
- Lowercasing optional (kept as-is for transformer tokenizers)
- HTML tag stripping
- URL → `[URL]` token
- Mention `@x` → `[MENTION]` token
- Trim to 512 wordpiece tokens for transformer compatibility

### Tokenization (`scripts/04, 05`)
- RoBERTa: `jcblaise/roberta-tagalog-base` tokenizer, max_len=256, padding=`max_length`
- DistilBERT: `distilbert-base-multilingual-cased` tokenizer, same settings

### Feature extraction (`scripts/06_extract_features.py`)
- CLS embedding extracted from final hidden layer of each frozen detector
- PCA to 64 components (preserves ~95% variance on the JCBlaise train set)
- Output tabular features: `{roberta_pca_*, distil_pca_*, p_roberta_fake, p_distil_fake}`

---

## §5 — Embedding workflow ("Dual-Embedding")

The "DE" in "DE-GNN" in this codebase refers to **Dual-Embedding**: concatenated PCA-reduced features from both RoBERTa-Tagalog and DistilBERT-multilingual.

```python
# scripts/06_extract_features.py — schematic
roberta_cls = roberta(input_ids, attention_mask)[0][:, 0, :]   # [N, 768]
distil_cls = distilbert(input_ids, attention_mask)[0][:, 0, :] # [N, 768]
roberta_pca = pca_roberta.transform(roberta_cls.numpy())       # [N, 64]
distil_pca = pca_distil.transform(distil_cls.numpy())          # [N, 64]
features = concatenate([roberta_pca, distil_pca, p_rob, p_dis]) # [N, 130]
```

**Why both:**
- RoBERTa-Tagalog: Filipino specialist; captures Tagalog morphology
- DistilBERT-multilingual: handles code-switched Taglish + English fragments
- Disagreement between them = uncertainty signal, useful for NEUTRAL cards

**Paper-to-code naming reconciliation:** the BATB paper §2.5 discusses "DE" as **Differential Evolution** (evolutionary hyperparameter optimization). In this codebase, "DE" operationally refers to **Dual-Embedding**. Differential Evolution optimizer is scaffolded as future work / Thesis 3 scope. The paper §5 Limitations subsection documents this reconciliation honestly.

---

## §6 — Graph generation

### kNN similarity graph (`scripts/09_train_degnn.py`)
- **Nodes:** all training examples + their dual-embedding feature vectors
- **Edges:** kNN with k=5 in cosine-similarity space over edge features (the `p_roberta_fake`, `p_distil_fake` columns)
- **Symmetric:** for each pair (i, j) found by kNN, add both i→j and j→i
- **Train/test isolation:** `StandardScaler.fit()` runs on TRAIN ONLY (no test leakage); at inference, new nodes attach via train→new directed edges, leaving train neighborhoods untouched

### Why kNN (not the real social network)?
- Social-network edges aren't available for the JCBlaise dataset
- kNN approximates "posts with similar credibility signals cluster"
- GraphSAGE aggregates neighborhood signals → smoother, more robust predictions than per-instance classification

---

## §7 — Model architecture

### Detectors
- **RoBERTa-Tagalog**: 12-layer base, 768-dim hidden, 110M params. Fine-tuned with AdamW (lr=2e-5), warmup, 3 epochs, EarlyStoppingCallback on val_f1.
- **DistilBERT-multilingual**: 6-layer, 768-dim hidden, 67M params. Same training recipe.
- **5 prime seeds:** [13, 29, 47, 89, 127] per Liu et al. 2019. Best seed selected via validation F1; test F1 reported.

### GraphSAGE (`scripts/minerva_degnn.py`)
```python
class GraphSAGE(nn.Module):
    """Two-layer GraphSAGE with mean aggregator."""
    def __init__(self, in_dim, hidden=64, out_dim=2, dropout=0.2):
        # Layer 1: in_dim → hidden
        # Layer 2: hidden → hidden  
        # Output: hidden → out_dim (binary)
```
- **Aggregator:** mean over kNN neighbors
- **Training:** 60 epochs, Adam (lr=1e-3), EarlyStopping
- **Single-seed result** at present; multi-seed deferred to publication

### QLattice (`scripts/08_train_qlattice.py`)
- Symbolic regression via `feyn` library
- Input features: PCA features + detector probabilities + DE-GNN logit
- Output: compact equation stored as text in `models/qlattice_equation.txt`
- Used downstream for:
  - Difficulty tier inference (`|p - 0.5|` = margin)
  - Interpretable scoring (the Unity game will surface this as a meter)
  - Filter-cascade rejection (score must clear threshold)

### GPT-2 generator (`scripts/11b_train_gpt2_neurosymbolic.py`)
- Base model: `jcblaise/gpt2-tagalog`
- **18 control tokens** (Keskar CTRL-style):
  - `<|label=real|>`, `<|label=fake|>`
  - `<|graph=high|>`, `<|graph=mid|>`, `<|graph=low|>`, `<|graph=unk|>`
  - `<|qlat=high|>`, `<|qlat=mid|>`, `<|qlat=low|>`, `<|qlat=unk|>`
  - `<|ensem=high|>`, `<|ensem=mid|>`, `<|ensem=low|>`, `<|ensem=unk|>`
  - `<|tier=novice|>`, `<|tier=proficient|>`, `<|tier=advanced|>`, `<|tier=unk|>`
- **Percentile binning** (33/33/33 split per axis) — fixes the dominant-bin issue from earlier versions
- **Training seed:** 1729 (Picard 2021 compliant — avoids the cherry-picked 42)
- **Training:** 3 epochs, AdamW (lr=5e-5), batch_size=4 on A100

---

## §8 — Training configuration

### Hardware
- **Recommended:** Colab A100 + High-RAM (~30 min full pipeline)
- **Fallback:** Colab T4 (~75 min)
- Batch sizes scale automatically based on detected GPU memory (`scripts/minerva_config.py`)

### Reproducibility (Pineau 2021 compliant)
- All seeds documented and called via `transformers.set_seed()`
- `reports/_environment.json` snapshots Python, torch, transformers, CUDA, host
- Pool hash + bank hash + run_id logged in every report
- Train-only `StandardScaler.fit()` — no test leakage
- Sealed test split; best-by-validation export

### Key hyperparameters (centralized in `scripts/minerva_config.py`)
```python
DETECTOR_SEEDS = [13, 29, 47, 89, 127]   # Liu 2019 protocol
GPT2_TRAIN_SEED = 1729                    # Picard 2021
DETECTOR_LR = 2e-5
DETECTOR_EPOCHS = 3
GPT2_LR = 5e-5
GPT2_EPOCHS = 3
KNN_K = 5
GRAPHSAGE_HIDDEN = 64
GRAPHSAGE_EPOCHS = 60
PCA_COMPONENTS = 64
POOL_TARGET_SIZE = 668
DECK_SIZE = 56
NUM_USER_DECKS = 8
```

---

## §9 — Evaluation methodology

### Detector evaluation (`scripts/15_evaluate_detectors.py`)
- Per-seed test F1 (binary classification, macro-averaged)
- Reported as **mean ± std** across 5 prime seeds
- Best-by-validation seed exported for downstream use
- Calibration: softmax of logits used as probability; no temperature scaling currently applied

### Pipeline-level evaluation
| Report | What it measures |
|---|---|
| `reports/faith.json` | Faithfulness: % of cards whose indicator phrases match canonical bank entries |
| `reports/strict_allowlist.json` | Strict allowlist: % of cards whose entities pass the safety gate |
| `reports/balance.json` | Schema-invalid drops (should be 0) + bucket fill rates |
| `reports/pool.json` | Final pool composition + explanation diversity |
| `reports/pseudo.json`, `reports/pseudo_places.json` | Counts of pseudonym replacements |
| `reports/draw.json` | 8-deck overlap statistics |
| `reports/degnn_report.json` | GraphSAGE single-run results |
| `reports/_environment.json` | Pineau 2021 environment capture |

### Headline targets (paper §1.4 + §3.5)
- Faithfulness ≥98% → **actual 100.0%**
- Strict allowlist ≥99% → **actual 99.09%**
- RoBERTa F1 ~95.6% → **actual 95.30% ± 0.40%**
- DistilBERT F1 ~91.0% → **actual 91.73% ± 0.65%**
- Schema drops ≈ 0 → **actual 0**

---

## §10 — Inference pipeline

For demonstrating live inference (e.g., to a panelist):

```python
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load best-by-validation RoBERTa
ROBERTA_DIR = Path("models/roberta_finetuned")
tok = AutoTokenizer.from_pretrained(ROBERTA_DIR)
model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_DIR)
model.eval()

# Predict on a Tagalog post
text = "SHARE NOW BAGO MABURA: Kakaibang ginagawa ng kandidato A."
inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    logits = model(**inputs).logits
p_fake = torch.softmax(logits, dim=-1)[0, 1].item()
print(f"p(fake) = {p_fake:.4f}")
```

For DistilBERT and the ensemble, load from `models/distilbert_multilingual_finetuned/` and average the two probabilities (or use the QLattice equation in `models/qlattice_equation.txt`).

---

## §11 — Important helper modules

| Module | Purpose |
|---|---|
| `scripts/minerva_config.py` | Centralized hyperparameters + GPU-aware batch sizing |
| `scripts/minerva_schemas.py` | Pydantic models for every JSON output (extra="forbid") |
| `scripts/minerva_indicators.py` | 12-indicator definitions (EMO, URG, ANON, MISS, FAB, POL, CONS, DISC, IMP, REV, ENDO, RECF) + SIFT moves |
| `scripts/minerva_response_bank.py` | TL + EN phrase bank lookup + version reconciliation |
| `scripts/minerva_candidates.py` | Fictional candidate profiles (Candidate A/B/C) |
| `scripts/minerva_privacy.py` | NER + blocklist for person pseudonymization |
| `scripts/minerva_filters.py` | Multi-stage acceptance filters |
| `scripts/minerva_degnn.py` | GraphSAGE implementation (no PyG dependency) |
| `scripts/minerva_qlattice.py` | QLattice equation parsing + evaluation |

### Pydantic schemas (`scripts/minerva_schemas.py`)
All JSON outputs are validated against pydantic models with `extra="forbid"`. This means:
- Any unknown field anywhere in the pipeline raises a clear error
- Schema drift is impossible without an explicit code change
- A failing card surfaces immediately, not 5 stages downstream

The IndicatorPhrase schema added in v2.9.6 unlocked GPT-2 cards to flow through the pipeline (previously rejected en masse due to a missing optional field).

---

## §12 — Reproducibility instructions

### Local test suite (no GPU)
```bash
git clone -b main https://github.com/robertgeraldsenasin/MINERVA.git
cd MINERVA
python -m venv .venv && source .venv/bin/activate
pip install -r dev-requirements.txt
python -m pytest tests/ -q
# Expected: 311 passed, 1 skipped (~3s)
```

The 1 skipped test is `test_degnn_graph.py` skipping cleanly because torch isn't in dev-requirements. Install `requirements.txt` to run it.

### Full pipeline (Colab A100 + High-RAM)
1. Open Colab → File → Open notebook → GitHub tab
2. Repo: `robertgeraldsenasin/MINERVA` · Branch: `main`
3. Open `notebooks/MINERVA_Run_Colab_v2.9.6.ipynb`
4. Runtime → Change runtime type → **A100 GPU** + **High-RAM**
5. Runtime → Run all (~30 min)
6. Defense-grade backup is the second-to-last cell; model artifacts is the last cell

### Post-merge tail re-run (after a small code change, no GPU)
```bash
python scripts/35_pseudonymize_places.py [...]
python scripts/21_balance_unity_cards.py [...]
python scripts/23_enforce_election_theme.py [...]
python scripts/24_curate_teaching_cards.py [...]
python scripts/33_strict_name_allowlist.py [...]
python scripts/26_faithfulness_audit.py [...]
python scripts/28_draw_user_deck.py [...]
```
~5 minutes total; models are cached.

---

## §13 — Environment setup

```bash
python --version    # 3.11 or 3.12 required
```

`requirements.txt` pins:
- torch>=2.0
- transformers>=4.35
- datasets
- scikit-learn
- pandas, numpy<3.0
- feyn (QLattice)
- pydantic>=2.0

`dev-requirements.txt` for testing only:
- pytest
- pydantic
- scikit-learn
- numpy<3.0

The full pipeline requires torch + transformers (~3 GB install on Colab). Local dev only needs the dev-requirements (~150 MB).

---

## §14 — Experiment tracking conventions

- **No external tracker** (no W&B, no MLflow). All experiment results live in `reports/*.json` with hashed provenance.
- **Run identification:** every report carries `ts` (ISO timestamp) + `run_id` (8-char hex) + relevant version stamps.
- **Comparing runs:** download two run zips from Colab and diff `reports/*.json`. The reports are designed to be diff-friendly.
- **Reproducing a specific run:** check the `_environment.json` from that run for Python/torch/transformers versions; pip install those exact versions; clone the repo at the corresponding tag.

---

## §15 — Final-state summary

- **Code version:** v2.9.9 (final code release)
- **Tests:** 311 passed, 1 skipped (clean torch skip)
- **Faithfulness:** 100% (658/658)
- **Strict allowlist:** 99.09% (652/658; 1 correct safety block + 5 generic terms documented)
- **Detectors:** RoBERTa 95.30% ± 0.40% / DistilBERT 91.73% ± 0.65% (5 prime seeds, paper targets met)
- **Pool:** 658 cards (90 GPT-2 + 568 templates), 8 user decks × 56 cards
- **Reproducibility:** Pineau 2021 + Liu 2019 + Picard 2021 compliant
- **Repository structure:** clean (legacy isolated, README current, this codebook centralizes methodology)
- **Defense window:** November 18-22, 2026 (FEU Thesis 1 schedule)

---

## §16 — Cross-references

- For a 5-minute panelist briefing: read `docs/architecture.md`
- For version history: read `CHANGELOG.md`
- For the holdout validation strategy decision: read `docs/HOLDOUT_VALIDATION_STRATEGY.md`
- For the BATB paper alignment story: read `docs/V2.9.0_AUDIT_RESPONSE.md`
- For paper-to-code naming reconciliation (DE-GNN): see §5 above + README's "A note on the 'DE-GNN' naming"

---

## §17 — Notebook walkthrough (what each section does)

The canonical notebook is `notebooks/MINERVA_Run_Colab.ipynb`. It runs end-to-end on Colab A100 + High-RAM in ~30 minutes. Section-by-section purpose:

### Section 1 — Configuration
Sets pipeline mode (`PIPELINE_MODE`), GPT-2 flag (`USE_GPT2`), repo URL, and branch. Edit these constants only if you need a different run mode (the defaults are correct for a full retrain).

### Section 2 — Mount Google Drive (optional)
Mounts Drive for backup persistence. Skipped gracefully if Drive auth fails or you're running without it.

### Section 3 — Clone the repo
Shallow-clones the production branch into the Colab workspace.

### Section 4 — Verify required files arrived
Sanity checks that the response bank, blocklists, and candidate profiles are present after clone.

### Section 5 — Install dependencies
Single `pip install -r requirements.txt`. Takes ~30 seconds on Colab A100.

### Section 6 — Working folders
Creates `data/`, `models/`, `generated/`, `reports/`, `logs/` if absent.

### Section 6b — Environment capture
Writes `reports/_environment.json` snapshotting Python, torch, transformers, CUDA, host. Pineau (2021) compliance.

### Section 7 — Run unit tests
Smoke-checks the codebase before running the pipeline. 311 tests in ~3 seconds. Fail-fast.

### Section 7a — Pre-flight dataset download verification
Runs script 01 to fetch the JCBlaise dataset and verifies its arrival.

### Section 7b — Detector training pipeline
Stages 7b.1 → 7b.6 train and evaluate both detectors plus secondary models:
- **7b.1:** scripts 01/02/03 — data preparation and stratified split
- **7b.2:** script 17 — 5-prime-seed training of RoBERTa + DistilBERT
- **7b.2.1:** seed-statistics summary (mean ± std)
- **7b.3:** script 06 — feature extraction (CLS + PCA)
- **7b.4:** script 08 — QLattice symbolic regression
- **7b.5:** script 09 — GraphSAGE training
- **7b.6:** script 15 — JCBlaise test-set evaluation, best-by-val export

### Sections 8 → 8b → 8b.5 — Card generation
- **8:** script 30 generates template cards deterministically
- **8b (optional, gated on `USE_GPT2`):** scripts 10b/11b/12b/13 — neuro-symbolic GPT-2 fine-tune + generation + QLattice scoring
- **8b.5:** script 29 — merge GPT-2 candidates into the template stream

### Sections 9 → 9b → 10 → 11 → 12 — Curation
- **9:** script 31 — person-name pseudonymization
- **9b:** script 35 — place-name pseudonymization (260+ blocklist)
- **10:** script 21 — balance verdicts × candidates × indicators; pydantic schema validate
- **11:** script 23 — reject off-theme content
- **12:** script 24 — final teaching-card curation

### Sections 13 → 14 → 15 → 16 — Safety chain + evaluation
- **13:** script 28 — draw 8 user-specific 56-card decks
- **14:** script 26 — faithfulness audit (re-extracts indicators, asserts set equality)
- **15:** script 32 — detector validation on templates (internal consensus metric)
- **16:** script 33 — STRICT ALLOWLIST ENFORCER (last line of defense)

### Section 16b — Optional held-out detector evaluation
Deferred to external Filipino fact-checker validation per `docs/HOLDOUT_VALIDATION_STRATEGY.md`. Kept here as a scaffold; default `--use_internal_pseudo_labels=False`.

### Section 16c → 17 → 17b — Export and verification
- **16c:** script 40 exports the pilot pack (HTML + CSV + JSON)
- **17:** final-dashboard cell — reads all reports and prints headline metrics
- **17b:** asserts every paper success criterion (faithfulness ≥98, allowlist ≥99, schema drops = 0)

### Sections 18 → 19 → 20 — Samples and backup
- **18:** displays 6 sample cards from a user deck
- **19:** previews 3 cards from the pilot pack
- **20:** saves all outputs to Drive (or downloads as zip if Drive unavailable)

### Optional: Refresh JCBlaise blocklist
Run-once cell that re-extracts surface forms from the JCBlaise dataset and rebuilds `templates/jcblaise_real_names_blocklist.txt`. Only needed when the upstream dataset changes.

---

## §18 — Defense-day demo flow

For a panel demo without re-running the full 30-minute pipeline:

1. **Live inference example** — load a saved best-by-val detector and run `model(text)` on a sample post (see §10 inference template).
2. **Open `reports/faith.json`** — show the `pass_rate: 100.0` line.
3. **Open `reports/strict_allowlist.json`** — show `pass_rate_pct: 99.09`.
4. **Open `reports/detectors_5seed_summary_v28_panel.json`** — show the 5-seed mean ± std.
5. **Open one card from `generated/unity_cards_pool_strict.json`** — show its verdict + indicators + Tagalog explanation + verifier action.
6. **Run `pytest tests/ -q`** — 311 tests in 3 seconds.

If a panelist asks how the system catches real political names, point to the 6 rejected cards in `reports/strict_allowlist.json` — 1 of them is "pacquiao" (a real PH senator), correctly blocked by the safety chain.

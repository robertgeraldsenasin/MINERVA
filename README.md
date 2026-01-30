# MINERVA (Fake News / Misinformation Detection Pipeline)

Hey! This repo is our **MINERVA** project: a *reproducible* pipeline for detecting fake news / misinformation using:

- **RoBERTa (Tagalog specialist)**  
- **DistilBERT (multilingual / Tagalog + English + code-switch-friendly)**  
- **Feature extraction** (embeddings + probabilities + simple text stats)  
- **Random Forest** (strong tabular baseline)  
- **Qlattice** (turns features into an interpretable “equation-style” rule)  
- **DE‑GNN (text graph model)** (optional “reasoning over similarity” layer)


---

## What you need installed

- **Python 3.10+** (3.10 recommended because it’s stable with the ML stack we pinned)
- (Optional) **NVIDIA GPU + CUDA PyTorch** if you want training to run fast.

---

## Setup (one-time)

### 1) Create and activate your virtual environment
**Windows PowerShell:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3) Quick sanity check
```powershell
python -c "import transformers, datasets, sklearn; print('transformers', transformers.__version__); print('datasets', datasets.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
```

---

## How the scripts connect (the “pipeline map”)

Think of the repo like a mini factory:

```
01_download_dataset.py
      ↓
02_prepare_dataset.py
      ↓
03_split_dataset.py
      ↓
04_train_robertaMINERVA.py      05_train_distilbertMINERVA.py
             ↓                             ↓
                06_extract_features.py (uses BOTH models)
                             ↓
07_train_random_forest.py   08_train_qlattice.py   09_train_degnn.py
```

Each script **expects files created by the previous scripts**.

---

## Running the full pipeline (01 → 09)

From the project root:

```powershell
python scripts\01_download_dataset.py
python scripts\02_prepare_dataset.py
python scripts\03_split_dataset.py

python scripts\04_train_robertaMINERVA.py
python scripts\05_train_distilbertMINERVA.py

python scripts\06_extract_features.py
python scripts\07_train_random_forest.py
python scripts\08_train_qlattice.py
python scripts\09_train_degnn.py
```

---

## Folder guide (what goes where)

These folders are **generated locally** when you run scripts:

- `data/raw/`  
  Downloaded datasets (CSV/ZIP extract). **Not committed** to GitHub.
- `data/processed/`  
  Cleaned + merged data + train/val/test splits. **Not committed**.
- `data/features/`  
  Embeddings, tabular features, DE‑GNN predictions. **Not committed**.
- `models/`  
  Trained models (RoBERTa, DistilBERT, RF, PCA, DE‑GNN). **Not committed**.
- `logs/`  
  Training logs and evaluation summaries (some small ones can be committed).
- `generated/`  
  Exports for Unity (JSON) and other final artifacts (usually not committed unless it’s a tiny example).

---

# What each file does (and what you can tweak)

## `requirements.txt`
**What it does:** pins versions so installs don’t randomly break.

**Why pinned?**
- `numpy<2` avoids annoying `datasets/arrow` “copy” errors.
- `transformers`, `datasets`, `accelerate` are pinned to avoid version mismatches.

**Common tweaks:**
- If you have GPU, install CUDA PyTorch separately (PyTorch has different wheels).

---

# Script Files:

## `scripts/01_download_dataset.py`
**What it does:**
- Downloads datasets (mainly via Hugging Face).
- If a dataset needs extra loader code (like SEACrowd), this script can fall back to a ZIP download.

**Outputs:**
- `data/raw/*.csv`
- `data/raw/seacrowd_ph_fake_news_corpus_zip/...`

**Common tweaks:**
- Change dataset IDs (top of file).
- If SEACrowd HF loading fails, you can install `seacrowd` and retry (optional).
- If ZIP URL changes, update the fallback URLs.

**Datasets used:**
- Fake News Filipino dataset (Tagalog) [1]
- WELFake dataset (English) [2]
- SEACrowd PH fake news corpus wrapper [3] + underlying corpus repo [4]

---

## `scripts/02_prepare_dataset.py`
**What it does:**
- Loads raw CSVs and converts everything into a **single clean format**:
  - `id`
  - `dataset`
  - `lang`
  - `text`
  - `label`

**Label convention (super important):**
We try to unify everything as:
- `label = 1` → **FAKE / NOT‑CREDIBLE**
- `label = 0` → **REAL / CREDIBLE**

**Outputs:**
- `data/processed/corpus.csv`

**Common tweaks:**
- If you discover a dataset’s label meaning is reversed, change the mapping constants at the top.
- Adjust the WELFake downsampling size if training is too slow on CPU.

---

## `scripts/03_split_dataset.py`
**What it does:**
- Splits `corpus.csv` into:
  - `train.csv`
  - `val.csv`
  - `test.csv`
- Uses stratification so your label balance doesn’t get wrecked.

**Outputs:**
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

**Common tweaks:**
- Change split ratios (train/val/test).
- Change stratification rule if you want to stratify by dataset+label instead of label-only.

---

## `scripts/04_train_robertaMINERVA.py`
**What it does:**
- Fine-tunes **RoBERTa Tagalog** (`jcblaise/roberta-tagalog-base`) for 2‑class fake news detection. [7]

**Expected behavior:**
- This is your **Tagalog specialist** model.

**Outputs:**
- `models/roberta_finetuned/`

**Common tweaks:**
- `FILTER_LANG`: train RoBERTa on Tagalog-only (recommended) or allow mixed.
- `MAX_LEN`: 256 is safe; increase if you have GPU + RAM.
- `per_device_train_batch_size`: increase if GPU memory allows.

**Paper reference:** RoBERTa [5]

---

## `scripts/05_train_distilbertMINERVA.py`
**What it does:**
- Fine-tunes **DistilBERT multilingual** for the same fake/real classification task. [8]

**Expected behavior:**
- This is your **multilingual / code-switch friendly** model.

**Outputs:**
- `models/distilbert_multilingual_finetuned/`

**Common tweaks:**
- `MAX_TRAIN_SAMPLES`: cap it if CPU training is too slow.
- Try different multilingual bases later if you want (but keep it documented).

**Paper reference:** DistilBERT [6]

---

## `scripts/06_extract_features.py`
**What it does:**
Turns your two fine-tuned Transformers into **feature generators**:

For each article:
- RoBERTa CLS embedding + P(fake)
- DistilBERT CLS embedding + P(fake)
- PCA compression (so features become manageable)
- Simple lexical stats (length, punctuation, digit ratio, etc.)

**Outputs:**
- `data/features/*_tabular.csv`
- `data/features/*_embeddings.npz`
- `models/pca_roberta.joblib`
- `models/pca_distilbert.joblib`

**Common tweaks:**
- `PCA_COMPONENTS`: more dims = better accuracy sometimes, slower + bigger RF/GNN.
- `BATCH_SIZE`: increase if you have GPU.
- Add more “cheap” features (caps ratio, emoji count, code-switch ratio).

---

## `scripts/07_train_random_forest.py`
**What it does:**
Trains a **Random Forest** on the tabular features from script 06.

**Why RF?**
- Strong baseline
- Easy to debug
- Feature importance is useful for analysis

**Outputs:**
- `models/random_forest.joblib`
- `logs/random_forest_report.txt` (if enabled)

**Common tweaks:**
- `n_estimators`: higher = better stability but slower.
- You can try XGBoost/LightGBM later (optional extension).

**Paper reference:** Random Forests (Breiman, 2001) [9]

---

## `scripts/08_train_qlattice.py`
**What it does:**
Fits a **Qlattice** model (symbolic regression) on the SAME feature table, producing an interpretable equation.

**Outputs:**
- `models/qlattice_equation.txt`
- (Sometimes) `logs/qlattice_notes.txt` if install/import fails

**Common tweaks:**
- If `feyn` install fails on your machine, you can skip this script and still finish the pipeline.
- Reduce features if Qlattice is too slow (stick to PCA + probabilities + a few lexical stats).

**Paper reference:** Qlattice / symbolic regression approach [10]

---

## `scripts/09_train_degnn.py`
**What it does:**
Creates a graph of articles:
- Nodes = articles
- Edges = kNN similarity in feature space (cosine similarity)

Then trains a GraphSAGE‑style GNN over that graph.

**Outputs:**
- `models/degnn.pt`
- `data/features/degnn_preds.csv`

**Common tweaks:**
- `KNN_K`: more neighbors = denser graph (can help but slower).
- `MAX_NODES`: lower this if your PC struggles.
- `EPOCHS`, `HIDDEN`: tuning knobs for model capacity.

**Paper references:**
- GraphSAGE idea (Hamilton et al.) [11]
- Rumor / fake info with GNNs (example: BiGCN AAAI) [12]

---

## Coming next (GPT module)

- `10_prepare_gpt2_corpus.py`  
- `11_train_gpt2.py`  
- `12_generate_validate_export.py`  

The idea is:
- GPT generates candidate “news cards”
- Your detection pipeline validates them
- Only validated content is exported to Unity

---

# Common issues (quick fixes)

### “CUDA is False but I have a GPU”
You installed CPU-only PyTorch. Install a CUDA PyTorch wheel (depends on your GPU + CUDA version).

### “SEACrowd requires dependency ‘seacrowd’”
Totally normal. Either:
- install `seacrowd` (optional), OR
- keep using the ZIP fallback included in script 01.

### “Training is slow”
- Reduce dataset size (downsample WELFake in script 02 or cap in script 05).
- Lower `MAX_LEN` (256 is already decent).
- Use GPU CUDA PyTorch.

---

# References

**Datasets**
1. Fake News Filipino (Hugging Face dataset card): https://huggingface.co/datasets/jcblaise/fake_news_filipino  
2. WELFake (Hugging Face dataset card): https://huggingface.co/datasets/davanstrien/WELFake  
3. SEACrowd PH fake news corpus (HF dataset card): https://huggingface.co/datasets/SEACrowd/ph_fake_news_corpus  
4. Philippine Fake News Corpus (source repo): https://github.com/aaroncarlfernandez/Philippine-Fake-News-Corpus  

**Models / Methods**
5. RoBERTa Tagalog base model (HF model card): https://huggingface.co/jcblaise/roberta-tagalog-base  
6. DistilBERT multilingual cased (HF model card): https://huggingface.co/distilbert/distilbert-base-multilingual-cased  
7. Breiman (2001). *Random Forests.* Machine Learning 45, 5–32. (Classic RF paper)  
8. Broløs et al. (2021). *An Approach to Symbolic Regression Using Feyn.* arXiv:2104.05417 — https://arxiv.org/abs/2104.05417  


---

## License / ethics note
This repo stores scripts, not the raw datasets. Always follow each dataset’s terms before redistributing any text content.

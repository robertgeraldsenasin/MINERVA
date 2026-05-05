# JCBlaise Fake News Filipino - Dataset Card for M.I.N.E.R.V.A.

> This document fulfills the user's request to "find and store the dataset of JC Blaise that we are using for this study, so we can adjust the censorship and filtration part of the module."

## Source

- **Dataset name:** Fake News Filipino
- **Authors:** Jan Christian Blaise Cruz, Julianne Agatha Tan, Charibeth Cheng (De La Salle University)
- **HuggingFace ID:** [`jcblaise/fake_news_filipino`](https://huggingface.co/datasets/jcblaise/fake_news_filipino)
- **Publication:**
  > Cruz, J. C. B., Tan, J. A., & Cheng, C. K. (2020). Localization of Fake News Detection via Multitask Transfer Learning. *Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020)*, 2596–2604. <https://aclanthology.org/2020.lrec-1.316/>

## Schema

| Field | Type | Description |
|---|---|---|
| `article` | string | Full text of the news article (mostly Tagalog, some Tagalog-English code-switching) |
| `label` | int (0 / 1) | 0 = REAL (factual reporting), 1 = FAKE (misinformation/disinformation) |

The dataset is published as a single set of records. `scripts/03_split_dataset.py` splits it 70/15/15 into train/val/test with a fixed seed for reproducibility (per BATB section 3.6.3).

## Why this dataset

Per the thesis paper section 1.5 Scope:

> "The system covers text-based social media posts and short news-like snippets derived from patterns of The JCBlaise Dataset. The JCBlaise dataset encompasses news articles containing misinformation written in Tagalog and English, and is labeled on a scale of 0 to 1, where 0 represents factual information and 1 represents misinformation."

This is the only Philippine-localized fake-news dataset of sufficient size that explicitly labels article-level credibility in Tagalog. Inclusion is one of the stated **delimitations** (section 1.5 Limitation #1).

## How M.I.N.E.R.V.A. uses it

1. **Detector training** (scripts 04 / 05 / 09 / 16 / 17). RoBERTa-Tagalog, DistilBERT-multilingual, and DE-GNN are fine-tuned on the train split for binary REAL/FAKE classification. Frozen test-split metrics: RoBERTa F1 = 95.6%, DistilBERT F1 = 91.0%, DE-GNN F1 = 95.8%.

2. **Feature extraction** (script 06). Pooled CLS embeddings from RoBERTa and DistilBERT pass through PCA. The resulting compact features (`r_pca_*` and `d_pca_*`) plus the detector probabilities (`p_roberta_fake`, `p_distil_fake`) become inputs to the Random Forest baseline (script 07), the QLattice symbolic regression (script 08), and the DE-GNN GraphSAGE network (script 09).

3. **GPT-2 conditioning corpus** (script 10). When `USE_GPT2=True`, JCBlaise articles become **control-token-conditioned** training examples for GPT-2-Tagalog. Each line in the corpus has the form:

   ```
   <|label=fake|> <|graph=high|> <article text>
   ```

   The `<|label=...|>` token comes from the article's true label. The `<|graph=high|mid|low|>` token is binned from DE-GNN's `p_degnn_fake` confidence on that article (output of script 09). GPT-2 fine-tunes on this corpus and learns a joint distribution `p(text | label, graph_confidence)`.

4. **GPT-2 generation** (script 12). Generation is prefixed with the desired `(label, graph)` pair. Each generated candidate is then scored by RoBERTa, DistilBERT, and DE-GNN. The `--accept_mode ensemble3` gate requires the average probability to exceed `min_conf` for the target label. **All three algorithms must agree** before a candidate is accepted.

5. **QLattice scoring** (script 13). Each accepted candidate's features are passed through the symbolic regression equation from script 08 (e.g., `logreg(0.371*dpca1 + 0.339*rpca0 + 0.852*exp(0.533*dpca3) - 0.69)`) to produce the final `p_fake_qlattice` score and a `margin` (distance from 0.5) used as the `difficulty_bin` on the Unity card.

This conditioning-token-based fine-tuning is the paper's specified architecture. It preserves GPT-2's pretrained Tagalog knowledge by:

- Training on the same JCBlaise text the base model already understands - only with new prepended conditioning tokens.
- Concentrating the new learning in the embeddings of the new `<|label=...|>` and `<|graph=...|>` tokens.
- Using the ensemble3 gate to filter at output time, so the base model is never punished for retaining pretrained behavior.

## The name-leakage problem (the issue v2.6.final fixes)

JCBlaise contains **real Filipino political names** drawn from real PH news reporting between roughly 2016–2020. When GPT-2 is fine-tuned on these articles even via the conditioning approach, those names enter the generator's vocabulary. At generation time, the existing pseudonymizer (`scripts/22_pseudonymize_entities.py` + `minerva_privacy.py`) catches *most* but not all because:

1. Single-token surnames without titles ("Pacquiao endorsed...") slip past the regex's title+name and particle+name patterns.
2. The pseudonymizer is **allowlist-protecting** (preserves the candidate codes from being replaced) but not **allowlist-enforcing** (doesn't reject other names that pass the pattern).
3. GPT-2 will sometimes emit names the pseudonymizer's regex didn't anticipate - partial matches, embedded mentions, code-switching variants.

Per the user's audit:
> "I noticed that when generating, sure there are analysis and ways to tell the students what made it real or fake, it's not isolated into those 3 candidates, and variety of names appear which could cause confusion and also legality issues."

The legality concern is real. Naming a real political figure in fake-news training material - even with a disclaimer - exposes the project to potential defamation claims and undermines BATB section 1.5 Limitation #2. Pilan et al. (2022) on the TAB anonymization benchmark explicitly recommends multi-pass enforcement (regex + dictionary + allowlist) for this exact failure mode.

## v2.6.final solution: scripts 33 + 34 (NEW)

### scripts/34_extract_jcblaise_names.py

Downloads JCBlaise from HuggingFace, runs every article through name-detection patterns matching script 33's, and emits `templates/jcblaise_real_names_blocklist.txt` - every person-like name appearing at least 3 times. The file is one-line-per-name lowercase plain text, git-diffable.

Run on Colab (HF Hub is not reachable from local dev):
```bash
python scripts/34_extract_jcblaise_names.py \
  --out_file templates/jcblaise_real_names_blocklist.txt \
  --report_out reports/jcblaise_extraction.json \
  --min_count 3
```

### scripts/33_strict_name_allowlist.py

The actual enforcer. Runs as the **last stage** of card generation - after templates, pseudonymization, theme filter, curation, faithfulness audit, detector validation. Two modes:

- **`--mode reject`** (recommended for game export): drops any card whose text contains a name not on the codes-only allowlist (`Candidate A`, `Candidate B`, `Candidate C`).
- **`--mode redact`**: replaces foreign names with a placeholder ("[Iba pang tao]" by default) and keeps the card.

Detection uses three passes:

1. **Structural patterns**: title+name, person-particle+name, "according to" attributions, said+surname, comma attribution, multi-word names.
2. **Multiword spans**: catches multi-word names regardless of context.
3. **Direct blocklist scan**: any single-word entry from the blocklist file matching anywhere in the text - catches naked surnames like "Pacquiao".

Tested with 19 unit tests covering both modes, all three candidate codes, naked-surname catches, overlap deduplication, case sensitivity, and definite-non-name handling (Tagalog function words, Philippine place names, real fact-checker organization names). **19/19 passing.**

### Pipeline integration

The new pipeline order:

```
[Mode A]  30 (templates) -> 31 (pseudonymize) -> 21 (balance) -> 23 (theme)
                         -> 24 (curate) -> 28 (deck) -> 26 (faith) -> 32 (det)
                         -> 33 (STRICT ALLOWLIST, NEW) -> 40 (pilot pack)

[Mode B]  09 (DE-GNN) -> 10 (corpus w/ label+graph tokens) -> 11 (GPT-2 fine-tune)
                      -> 12 (generate w/ ensemble3 gate) -> 13 (QLattice score)
                      -> 18 (verdict explain) -> 21 -> 23 -> 24 -> 28 -> 26 -> 32
                      -> 33 (STRICT ALLOWLIST, NEW) -> 40 (pilot pack)
```

In **both modes**, script 33 is the final filter. Script 40 (pilot pack) reads from `unity_cards_pool_strict.json` so the printable pre-pilot pack is guaranteed codes-only.

## End-to-end verification (this build)

Mode A pipeline run against the v2.6.final repo with codes-only candidate config:

```
Pool size           : 668 cards (108 REAL / 500 FAKE / 60 UNCERTAIN)
Indicator coverage  : 12/12
Faithfulness audit  : 100.0%
Pairwise overlap    : 11.48% mean (range 1.79-25.0)
Detector ensemble   : 100.0%
STRICT ALLOWLIST PASS RATE: 100.0%
Cards rejected      : 0
```

Every card mentions only "Candidate A", "Candidate B", or "Candidate C". Real fact-checking organizations (Vera Files, Pulse Asia, Department of Education, COMELEC) appear where appropriate as legitimate verification sources.

Mode B is wired correctly per the paper but requires Colab GPU for actual GPT-2 fine-tuning and generation. Once executed, the same script 33 enforcer applies.

## Citations

### Censorship architecture

- **Cruz, J. C. B., Tan, J. A., & Cheng, C. K. (2020).** Localization of Fake News Detection via Multitask Transfer Learning. *LREC 2020.*
- **Pilan, I., Lison, P., Ovrelid, L., et al. (2022).** The Text Anonymization Benchmark. *Computational Linguistics, 48*(4), 1053–1101. - multi-pass anonymization rationale.
- **Yermilov, O., Raheja, V., & Chernodub, A. (2023).** Privacy- and Utility-Preserving NLP with Anonymized Data. *EACL 2023.*
- **Roozenbeek, J., & van der Linden, S. (2019).** *Palgrave Communications, 5*(1). - fictional examples in inoculation games.
- **Roozenbeek, J., & van der Linden, S. (2020).** *HKS Misinformation Review, 1*(8).
- **Arugay, A. A., & Baquisal, J. K. A. (2022).** *Pacific Affairs, 95*(3). - PH political archetypes.
- **Mendoza, R. U., Beja Jr., E. L., Venida, V. S., & Yap II, D. B. (2012).** *PPSJ, 33*(2). - PH dynasty surname concentration.
- **BATB_CompiledThesisPaper section 1.5 Limitation #2.**

### Conditioning architecture (Mode B GPT-2 path)

- **Hamilton, W. L., Ying, R., & Leskovec, J. (2017).** Inductive Representation Learning on Large Graphs. *NeurIPS 2017.* - GraphSAGE for DE-GNN.
- **Brolos, K. R., Machado, M. V., Cave, C., et al. (2021).** An Approach to Symbolic Regression Using Feyn. arXiv:2104.05417. - QLattice.
- **Wolf, T., Debut, L., Sanh, V., et al. (2020).** Transformers. *EMNLP Demos.* - HuggingFace.
- **LekshmiAmmal, H. R., et al. (2021).** Ensemble Transformer Model for Fake News Classification. *CLEF CheckThat! Lab.* - ensemble3 gate rationale.

## Ethics note

JCBlaise's articles are publicly available news, and the dataset is published under terms permitting research use. M.I.N.E.R.V.A. does not redistribute the dataset; it downloads from HuggingFace at training time. The blocklist generated by script 34 contains names extracted from the public dataset and used solely to **prevent** their reproduction in game output - privacy-protective by design.

The names appearing in the blocklist do **not** indicate any of the named individuals are themselves associated with disinformation. They are public figures whose names appear in the corpus and must therefore be excluded from the fictional game scenario per the thesis paper's stated limitations.

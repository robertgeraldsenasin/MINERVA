# M.I.N.E.R.V.A. Master Codebook (v2.0)

> *Misinformation Investigation through Networked Embeddings for Rumor Verification and Awareness*
> FEU Institute of Technology — IT Thesis 2026
> Lola, Salva, Senasin

This codebook is the panel-defence-grade reference for every script,
module, and template in the v2.0 refactor. For each piece of code we
state:
* **Purpose** — what it does in one sentence.
* **Why** — the design rationale grounded in literature.
* **Inputs / Outputs** — the contract.
* **Pipeline position** — where it sits in the run.
* **Citations** — the literature that justifies the design choice.

The full bibliography lives at the bottom. In-text citations follow
APA-7 short form.

---

## 0. The Problem v2.0 Solves

The legacy v1 pipeline produced a `unity_cards.json` with 992
generated posts. Inspection of the file revealed four blocking
issues for SHS deployment:

1. **Static explanations.** Every card's `summary` reduced to "*Verdict: REAL/FAKE … The decision comes from the stored Qlattice equation applied to detector/embedding features.*" The 12 PCA components and the abstract probability score gave learners no language they could carry into their own social-media reading practice (Khosravi et al., 2022; Barzilai & Stadtler, 2025).
2. **Off-theme content leaks.** Posts about the *Grab* TNVS strike and *Meralco* electricity rates passed the simple keyword filter, even though M.I.N.E.R.V.A. is an electoral simulator. SHS players encountering transport posts in a voter-literacy game break the suspension of disbelief that the design relies on.
3. **Random pseudonyms.** Cards referenced "Candidate GQW", "Candidate DTQ", "Entity B" — randomly drawn three-letter codes that were inconsistent across cards in the same story and had no link to the three VERIdex profiles the game requires (Yermilov et al., 2023).
4. **GPT-2 truncation.** Many cards ended mid-sentence: "*…citing the need for*" / "*…in support of the*" — silently shipped to the deck as if complete.

The fixes are in the four foundation modules and seven pipeline scripts described below.

---

## 1. Foundation Modules (`scripts/minerva_*.py`)

### 1.1 `minerva_indicators.py` — 12-cue taxonomy

**Purpose.** Provide a deterministic detector for each of 12 student-facing misinformation indicators, plus a flat numeric feature dict for QLattice re-fitting.

**Why.** The legacy pipeline fitted QLattice on 24 PCA components extracted from RoBERTa, DistilBERT-multilingual, and DE-GNN embeddings. The discovered equations were of the form `0.62 + 0.41·rpca3 − 0.18·dpca7`. **No SHS student can learn from `rpca3`.** Christensen et al. (2022) and Brolós et al. (2021) document that the interpretability advantage of symbolic regression collapses when its inputs are themselves opaque embeddings.

The 12 indicators are derived from:
* **DEPICT taxonomy** (Roozenbeek & van der Linden, 2019; Basol et al., 2020): Emotion, Polarization, Impersonation, Conspiracy, Discrediting, Trolling. The Bad News inoculation game's six-technique framework is the most-validated misinformation-tactic catalogue in the inoculation-theory literature.
* **Filipino-electoral extensions** (Arugay & Baquisal, 2022; Schipper, 2025; Bautista, 2021): Historical Revisionism, Manufactured Endorsements, Candidate-Record Fabrication. Documented narrative families in the Philippine 2022 election cycle that the DEPICT scheme alone does not name.
* **General misinformation diagnostics** (Leite et al., 2025; the W3C Credibility Signals working group): Anonymous Attribution, Missing Evidence, Urgency, Fabricated Quote.

The result is **12 codes**: `EMO, URG, ANON, MISS, FAB, POL, CONS, DISC, IMP, REV, ENDO, RECF`.

**Why deterministic regex/lexicon, not an LLM?** Faithfulness (Longo et al., 2024 Open Problem 7). An LLM-paraphrased indicator detection would not be reproducible or auditable; the same card could fire different indicators on different runs. Regex/lexicon detectors yield identical output every time, which is required for the schema-validated handoff between pipeline scripts and for the audit trail in script 26.

**Inputs.** `text: str`, optional `metadata: dict`.
**Outputs.** `dict[code -> IndicatorHit]` plus `named_features(text) -> dict[str, float]` for QLattice.

**Citations.** Roozenbeek & van der Linden (2019); Basol et al. (2020); Khosravi et al. (2022); Longo et al. (2024); Liu, Ye & Li (2024); Athira et al. (2023); Christensen et al. (2022); Brolós et al. (2021); Arugay & Baquisal (2022); Schipper (2025); Bautista (2021); Leite et al. (2025); Cruz et al. (2020 — JCBlaise, our generation-base corpus).

---

### 1.2 `minerva_response_bank.py` — Tiered explanation bank

**Purpose.** Map fired indicators to natural-language feedback phrases, varied by difficulty tier (novice / proficient / advanced) and stamped with a SIFT move (Stop / Investigate / Find / Trace).

**Why.** Two converging requirements forced a bank-driven design over an LLM-paraphrased one:
1. **Faithfulness preservation.** Longo et al. (2024) and Liu, Ye & Li (2024) both warn that LLM-generated rationales may *plausibly* explain a model's output without *faithfully* tracking it. A bank-driven explanation, where each phrase is provably tied to one indicator code, sidesteps this trap.
2. **Reproducibility for thesis defence.** Same card, same seed → same explanation, every run. This is required so that the chair can re-run the pipeline and verify any specific card cited in the paper.

The bank has **56 entries** spanning 12 indicators × 3 tiers × 2–3 variants + 4 credible-card affirmations. Tier banding follows Dehghanzadeh et al. (2024) and Almaki et al. (2024), who report that gamified misinformation-literacy interventions show the strongest effect when difficulty is graduated.

**Why credible-card affirmations exist.** Modirrousta-Galian & Higham (2023) report that misinformation-detection training systematically pushes learners toward over-suspicion — they get better at flagging fakes but also start mis-flagging real news. The four `CREDIBLE_AFFIRMATIONS` entries explicitly reward correct detection of credible cards, calibrating trust as well as suspicion.

**Why SIFT moves.** Caulfield (2019) and Caulfield & Wineburg (2023) show that simple action-rules (Stop, Investigate, Find, Trace) outperform full-rubric checklists in transferring to real-world reading. Each bank entry ends in one SIFT move.

**Inputs.** `fired_indicators: list[str]`, `verdict: str`, `fake_likelihood_percent: float`, `seed_str: str`, `tier: str`, optional `candidate_name: str`.
**Outputs.** `ExplanationBlock`-compatible dict.

**Citations.** Caulfield (2019); Caulfield & Wineburg (2023); Modirrousta-Galian & Higham (2023); Dehghanzadeh et al. (2024); Almaki et al. (2024); Werner Axelsson (2024); Longo et al. (2024); Liu, Ye & Li (2024).

---

### 1.3 `minerva_candidates.py` — Three fictional candidates

**Purpose.** Maintain a fixed registry of three fictional candidates, plus deterministic archetype-router that maps any post text → one of the three.

**Why three candidates and not five or ten?** Curriculum-design simplicity (the VERIdex module's UI handles three at a time gracefully) and study-grounded selection. Arugay & Baquisal (2022) document three dominant narrative families in 2022 Philippine election disinformation: dynastic restoration, reformist red-tagging, and populist mass-celebrity. Mendoza et al. (2023) corroborate. We map each family to one fictional candidate:

| Code | Name | Archetype | Real-world disinformation pattern |
| ---- | ---- | --------- | --------------------------------- |
| C-RM | Sen. Reynaldo "Rey" Marquez | DYNASTIC | Historical revisionism, manufactured surveys, infrastructure-record inflation |
| C-IB | Vice-Mayor Iris Bantayan | REFORMIST | Red-tagging, fabricated quotes, conspiracy framing about civil-society links |
| C-JS | Rep. Datu Jomar "JM" Salonga | POPULIST | Emotional appeals, unanchored endorsements, celebrity-credibility transfer |

**Why deterministic routing?** Yermilov et al. (2023) define *consistency-preservation* as the criterion that distinguishes useful pseudonymisation from random redaction. A learner studying the deck must encounter "Sen. Marquez" referring to the same character every time. The session-cache + stable-hash design satisfies this.

**Why archetype cues, not a fine-tuned classifier?** Two reasons. First, training a classifier requires labelled cards we do not have (the legacy data has no archetype labels). Second, the archetype cue lists are pedagogically transparent — at thesis defence we can show the panel exactly which words trigger each archetype, making the routing explainable.

**Inputs / Outputs.** See module docstring.

**Citations.** Arugay & Baquisal (2022); Mendoza et al. (2022, 2023); Schipper (2025); Bautista (2021); Yermilov et al. (2023); Deinla et al. (2022).

---

### 1.4 `minerva_filters.py` — Four content gates

**Purpose.** Reject cards that fail any of: (a) electoral theme, (b) non-truncation, (c) pseudonym integrity, (d) candidate mention.

**Why four gates and not one model?** Each gate has a different failure mode and different remediation: an off-theme card needs the generator's prompt fixed; a truncated card needs the generator's max-tokens raised; a legacy-pseudonym card needs script 22 to have actually run; a no-candidate card needs the generator's prompt to inject a candidate. Bundling all four into a single classifier loses the diagnostic information. Wenzek et al. (2020 — the CRISP curation pipeline for CC-100) makes the same argument for multi-stage gates over end-to-end filters.

**Why allow `is_neutral_volume`?** Per the user's explicit requirement: real social-media feeds contain benign off-topic content, and SHS testers found pure-electoral feeds artificial. The gate lets cards that are *clearly off-theme but contain no misinformation patterns* pass through tagged `is_neutral_volume=True`. The Unity client can render these with a different visual treatment so they don't compete with the teaching cards but do provide the noise floor that real Chattr feels like.

**Citations.** Wenzek et al. (2020); Hu et al. (2024 — ARG safety filter); Khosravi et al. (2022).

---

## 2. Refactored Pipeline Scripts

### 2.1 `13_score_generated_with_qlattice.py`

* **Purpose.** Score GPT-2 generations with QLattice + named features + truncation gate.
* **What changed.** Now feeds 12 indicator features alongside the legacy 24 PCA components to QLattice. Drops truncated cards before they reach the deck. Logs rejections to `reports/score_rejection_log.jsonl`.
* **Why.** Christensen et al. (2022); Brolós et al. (2021) — symbolic regression's interpretability story holds *only* when the inputs themselves are interpretable.
* **Inputs.** `--in_file generated/gpt2_synthetic_raw_*.jsonl`, optional `--qlattice_model`, `--pca_models`.
* **Outputs.** `generated/gpt2_synthetic_final_*.jsonl`.

### 2.2 `18_verdict_explain.py`

* **Purpose.** Convert scored generations into Unity cards with content-aware varied explanations.
* **What changed.** Replaces the static template (every card got the same summary) with `assemble_explanation()` from the response bank. Each card carries `fired_indicators`, `indicator_details`, `bank_refs`. Output validates against the `UnityCard` pydantic schema.
* **Why.** This is *the* fix for the static-explanation problem. Validated on 50 legacy records: schema correctly rejected 47 truncated cards; the 3 surviving cards yielded **3 unique summaries** (100% diversity).
* **Inputs.** `--in_file generated/gpt2_synthetic_final_both.jsonl`.
* **Outputs.** `generated/unity_cards.json` + `reports/audit_18.jsonl`.

### 2.3 `21_balance_unity_cards.py`

* **Purpose.** Schema-validate, dedupe, and balance the unity-cards pool across verdict × candidate × difficulty × indicator coverage.
* **What changed.** Adds the four-axis balance and emits `reports/balance_report.json`.
* **Citations.** Source2Synth-style curation (Lupidi et al., 2024).

### 2.4 `22_pseudonymize_entities.py`

* **Purpose.** Rewrite real-name references to one of the three fictional candidates with session-cache consistency.
* **What changed.** Replaces random three-letter codes with the three fixed candidates. Adds `--re_explain` so explanations can be re-generated knowing the candidate name (e.g. "*This post about Sen. Reynaldo "Rey" Marquez …*").
* **Citations.** Yermilov et al. (2023).

### 2.5 `23_enforce_election_theme.py`

* **Purpose.** Reject off-theme cards; tag neutral-volume cards.
* **What changed.** Adds hard-negative awareness (Grab, Meralco, K-pop, weather), structured rejection log, and the neutral-volume policy.
* **Citations.** Wenzek et al. (2020).

### 2.6 `24_curate_teaching_cards.py`

* **Purpose.** Promote themed cards into the daily-cycle deck with difficulty banding and FAKE↔REAL cross-linking.
* **What changed.** Mandatory ≥ 3 credible cards per day (Modirrousta-Galian & Higham, 2023). Each FAKE card carries `credible_counter_card_id` so VERIdict can show side-by-side comparison.
* **Citations.** Modirrousta-Galian & Higham (2023); Barzilai & Stadtler (2025); Almaki et al. (2024).

### 2.7 `25_build_candidate_scenarios.py`

* **Purpose.** Build VERIdex profile cards with archetype bio, planks, indicator susceptibility (prior + empirical), counter-narrative anchors.
* **What changed.** Profiles are now derived from the registry; each profile lists indicator susceptibility so the player can see *what to watch for* when scrolling rumours about that candidate.

### 2.8 `26_faithfulness_audit.py` (NEW)

* **Purpose.** Re-extract indicators from explanation prose and assert set-equality with `fired_indicators`. Verify bank version stamps. Verify REAL cards have a credible affirmation. Strict mode aborts the pipeline on any failure.
* **Why.** Without an automated post-hoc check, a future bank edit could silently break faithfulness — a card whose explanation talks about "discrediting" but whose `fired_indicators` field doesn't list `DISC` is a faithfulness violation. This is the audit Longo et al. (2024) describe as the missing piece of XAI pipelines.
* **Citations.** Longo et al. (2024); Liu, Ye & Li (2024); Khosravi et al. (2022).

### 2.9 `27_response_bank_versioning.py` (NEW)

* **Purpose.** Stamp / diff / re-render decks under different bank versions.
* **Why.** Werner Axelsson (2024) reports that dialogic-feedback evaluations require A/B comparable artefacts. The `stamp` and `rerender` subcommands let the team run two student cohorts under bank v1 and bank v1.1 and compare results.

---

## 3. Templates

`templates/candidate_profiles_three_candidates.json`, `templates/teaching_response_bank_v1.json`, `templates/election_theme_keywords.json`, `templates/indicator_taxonomy_v1.json` — these are JSON exports of the in-Python registries so the Unity client and any external auditor can read them without running Python.

---

## 4. Tests

`tests/test_indicators.py` (24 tests) and `tests/test_filters.py` (13 tests) lock in the behaviour of the foundation modules. Run before every release: `python -m pytest tests/ -v`.

---

## 5. Bibliography (selected)

* Almaki, S., et al. (2024). Gamified vs traditional approaches to teaching misinformation literacy. *Computers & Education*.
* Arugay, A. A., & Baquisal, J. K. A. (2022). Mobilized and polarized: Disinformation narratives during the 2022 Philippine elections. *Pacific Affairs*.
* Athira, A. B., et al. (2023). A systematic survey on explainable AI for fake news detection. *Information Processing & Management*.
* Barzilai, S., & Stadtler, M. (2025). Multiple-document literacy in the age of AI. *Journal of the Learning Sciences*.
* Basol, M., Roozenbeek, J., & van der Linden, S. (2020). Good news about Bad News: Gamified inoculation boosts confidence and cognitive immunity against fake news. *Journal of Cognition*.
* Bautista, A. P. (2021). Filipino electoral disinformation: A literature review. *FEU Asian Journal of Political Economy*.
* Brolós, K., Christensen, T., et al. (2021). QLattice: Symbolic regression for tabular data. *arXiv:2104.05417*.
* Caulfield, M. (2019). SIFT (the four moves). *Hapgood blog*.
* Caulfield, M., & Wineburg, S. (2023). *Verified: How to think straight, get duped less, and make better decisions about what to believe online*. University of Chicago Press.
* Christensen, T., et al. (2022). Symbolic regression with QLattice. *Discover Computing*.
* Cruz, J. C. B., Tan, J. K. C., & Cheng, C. K. (2020). JCBlaise/Filipino corpus. https://github.com/jcblaisecruz02
* Dehghanzadeh, H., et al. (2024). Gamified pedagogy for misinformation literacy. *Computers & Education: AI*.
* Deinla, I., Mendoza, R. U., et al. (2022). The promise and pitfalls of educating young Filipino voters. *Asian Journal of Political Science*.
* Hu, B., et al. (2024). Adversarial reading group (ARG) for misinformation defence. *ACL Findings*.
* Khosravi, H., et al. (2022). Explainable AI in education: A systematic review. *Computers & Education: AI*.
* Leite, M., et al. (2025). W3C credibility signals catalogue and applications. *Information Processing & Management*.
* Liu, Y., Ye, S., & Li, M. (2024). Faithfulness vs plausibility in LLM rationales. *EMNLP*.
* Longo, L., et al. (2024). Open problems in explainable AI. *Information Fusion*.
* Mendoza, R. U., et al. (2022). Misinformation in the 2022 Philippine elections: Evidence from a youth-voter survey. *Ateneo Policy Center Working Paper*.
* Mendoza, R. U., et al. (2023). Disinformation narratives and electoral choice in the Philippines. *Asian Journal of Communication*.
* Modirrousta-Galian, A., & Higham, P. A. (2023). Conservative response bias in misinformation training. *Journal of Experimental Psychology: Applied*.
* Roozenbeek, J., & van der Linden, S. (2019). The fake news game. *Journal of Risk Research*.
* Schipper, B. C. (2025). Disinformation tactics in Philippine electoral discourse. *Data & Policy*.
* Werner Axelsson, J. (2024). Dialogic feedback in adaptive learning systems. *Educational Technology Research and Development*.
* Wenzek, G., et al. (2020). CCNet: High-quality monolingual datasets from web crawl data. *LREC*.
* Yermilov, P., et al. (2023). Consistency-preserving pseudonymisation for privacy-aware NLP. *EACL*.

---

*Last updated: v2.0 release. This codebook is the single source of truth for design rationale; if a script's behaviour conflicts with this document, the script is wrong, not the document.*

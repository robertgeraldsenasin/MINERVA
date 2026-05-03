# Handoff to Claude Code — Open Work Items

> **From:** chat-side Claude (Anthropic web interface, working with Robert).
> **To:** Claude Code (the agent running locally on Robert's machine).
> **Status:** v2.6-final-consolidated is shipped and stable. Everything below is **future work** the user may or may not pursue.

---

## Priority 1 — defensible-for-thesis-defense

These are the items most likely to come up at the FEU IT thesis defense panel. If the user asks "make this more defensible" or "panel-prep," start here.

### P1.1 — Run a held-out detector evaluation

**The gap:** `scripts/32_validate_detectors_on_templates.py` validates synthetic scores from script 30, not a fresh inference from the trained models. A panel could ask "do your detectors generalize beyond your templates?"

**The work:**
1. Hold out 50 cards from the 668-card pool (stratified by tactic + verdict)
2. Run `scripts/15_evaluate_detectors.py` against them with the trained model weights loaded
3. Compare ground-truth detector scores against the synthetic scores in the held-out cards
4. Write the results into `docs/V2.6_CHANGES.md` and a new `reports/heldout_eval.md`

**Stretch:** also run the detectors against 10-20 cards from the original v2.5 pool (real news + GPT-2 augmentation) for a v2.5-vs-v2.6 distribution-shift comparison.

### P1.2 — Pre-pilot with 5 SHS students (qualitative)

**The gap:** No human has read the cards critically.

**The work:** This isn't a Claude Code task per se — it's a research task for the user. But Claude Code can help by:
1. Generating a **printable card pack** (HTML or PDF) of 30 randomly-sampled cards from `pool.json`
2. Building a **Google-Forms-shaped questionnaire** (markdown with the 5 questions per card: Is it FAKE/REAL/UNCERTAIN? Why? What tactic? Confidence? Would you share?)
3. Generating an **answer key** from the cards' fired_indicators

If the user says "build the pilot pack," start with `scripts/40_export_pilot_pack.py` (new file) that does all three.

### P1.3 — Inter-rater reliability on the indicator labels

**The gap:** The cards' `fired_indicators` were assigned by Claude (chat-side) when designing each template. A panel could ask "did anyone else verify these labels?"

**The work:**
1. Sample 50 cards from `pool.json`
2. Build a labeling spreadsheet for two raters (the user + one classmate or advisor) to independently mark which DEPICT indicators each card contains
3. Compute Cohen's κ between the two raters
4. Compare each rater's labels to the templates' declared `fired_indicators`
5. Report κ in `docs/V2.6_CHANGES.md` under a new "Inter-rater reliability" section

If the user says "build the IRR study," start with `scripts/41_build_irr_spreadsheet.py` (new file).

---

## Priority 2 — system robustness

These improve the system but don't change defense outcomes. Tackle if the user has time before defense.

### P2.1 — Detector invariance under candidate-name swaps

**The hypothesis to test:** If the user swaps Cruz → Mendoza in `candidate_config.py`, does the detector accuracy on the regenerated pool stay at 100%?

**The work:**
1. Run the smoke pipeline with default Cruz/Reyes/Garcia. Save metrics.
2. Edit `candidate_config.py` to use 5 alternative name sets (e.g., Mendoza/Santos/Flores). For each:
   - Run smoke pipeline
   - Save detector validation metrics
3. Verify all 5 swaps produce ≥98% detector accuracy with the same overlap/coverage targets.
4. Document in `docs/CANDIDATE_SWAP_INVARIANCE.md` (new file).

### P2.2 — Tier-balance optimization

**The gap:** Tier ratio is 35.9/31.7/32.3% but target is 40/35/25 (advanced is over-supplied because deepfake/impersonation/fact-checker are inherently advanced patterns).

**The work:** Either:
- (a) Add 2-3 more novice templates (simpler tactics like "all-caps share request" or "missing date-stamp"), OR
- (b) Re-tier one advanced template to proficient if the disinformation-pattern complexity allows it

**Don't blindly re-tier** — each tier change must be defensible against Modirrousta-Galian & Higham (2023) per-tier calibration arguments.

### P2.3 — Template variety expansion (lower pairwise overlap further)

**The gap:** Pairwise overlap is 11.48% mean. Target was <15% so we're already there, but each template currently has only 3 textual variants. With more variants per template, overlap drops further.

**The work:** For each of the 17 templates, add 2-3 more textual variants (different opening hooks, different platform names, different rhetorical tactics within the same indicator family). Don't change `fired_indicators` — only the surface text.

After expansion, re-run smoke. Pairwise overlap should drop to ~8% mean.

### P2.4 — Rejection-log analysis for theme filter

**The gap:** Theme filter rejects ~16 cards per pipeline run as "off-theme" or "no-candidate-mention." Are these legitimate rejects or false negatives?

**The work:**
1. Read `reports/theme_rej.jsonl` after a pipeline run
2. For each rejection, classify as: legitimate / false-negative / borderline
3. If >5% are false-negatives, adjust the keyword threshold or extend `ELECTORAL_POSITIVE` in `minerva_filters.py`
4. Document the analysis in `docs/THEME_FILTER_AUDIT.md`

---

## Priority 3 — Unity integration

**This is the bigger arc the user will need help with eventually.** The Python pipeline produces `pool.json`. The Unity game needs to consume it.

### P3.1 — JSON schema doc for Unity team

**The work:** Generate a JSON-schema-style doc (or actual JSON Schema file) for `pool.json` that a Unity dev can read without knowing Python:
- Field names and types
- Required vs optional
- Verdict / tier / archetype enums
- Example card

Put it at `docs/UNITY_INTEGRATION.md` and `schemas/pool_card.schema.json`.

### P3.2 — C# port of script 28 (deck draw)

**The gap:** `scripts/28_draw_user_deck.py` runs server-side currently. Eventually the Unity client needs to draw decks itself (offline play).

**The work:** Port the deck-drawing logic from `scripts/28_draw_user_deck.py` to a single C# class. Match the algorithm exactly:
- Deterministic per-user seed (hash of user_id)
- 7 days × 8 cards = 56 cards
- Min 3 credible (REAL) per day
- Pairwise overlap target <15%

Write tests against the same expected outputs.

### P3.3 — Unity scene wireframes

**Out of scope for code** — this is design. But Claude Code can scaffold:
- A `unity_scenes/` directory with markdown specs for each scene
- A `unity_assets_manifest.csv` listing what art/sound assets the scenes need

---

## Priority 4 — defense materials

If the user gets to defense prep:

### P4.1 — Generate a thesis-appendix PDF

**The work:** Build a Pandoc-driven LaTeX template that generates a thesis appendix from:
- `pool.json` → "Appendix A: Card Pool Sample" (30 random cards)
- `reports/faith.json` → "Appendix B: Faithfulness Audit Report"
- `reports/det.json` → "Appendix C: Detector Validation"
- `docs/V2.6_CHANGES.md` → "Appendix D: Architecture Evolution"

Put the template at `appendix/thesis_appendix.tex.j2` and a generator script at `scripts/50_build_appendix.py`.

### P4.2 — Defense slide deck (Beamer or PPT)

**The work:** A 12-slide deck covering:
1. Title + thesis statement
2. The problem (PH electoral disinformation)
3. The intervention (psychological inoculation)
4. The architecture (3 detectors + template generator + Unity game)
5. Why templates beat GPT-2 (the v2.6 pivot)
6. Why common-name candidates (Roozenbeek + Hainmueller)
7. The 17 templates and 12 DEPICT indicators
8. Detector accuracy (95.6 / 91.0 / 95.8 F1)
9. Faithfulness audit (100%)
10. Pre-pilot results (if P1.2 done)
11. Limitations (templates can repeat, Unity port pending, IRR pending if P1.3 not done)
12. Acknowledgments + Q&A

Generate as both Markdown (for the user to edit) and a Pandoc-Beamer compile script.

---

## Quick-task hopper

These are 10-30 minute tasks the user might pop in at any time. If the user says "do something quick," pick from here:

- [ ] Run the smoke pipeline and report the numbers
- [ ] Read 5 random cards from `pool.json` and flag any that don't read naturally
- [ ] Generate a CSV of all 17 templates' tactic / verdict / tier / fired_indicators
- [ ] Plot the indicator-coverage histogram from `reports/pool.json` and save as PNG
- [ ] Verify `tests/test_filters.py` covers all 17 tactics (currently it covers ~6 — add coverage)
- [ ] Run `python scripts/32_validate_detectors_on_templates.py` and append the numbers to `docs/V2.6_CHANGES.md`
- [ ] Lint `scripts/30_template_scenario_generator.py` for unused imports / long lines
- [ ] Add `scripts/__init__.py` if it's missing (some IDEs need it)

---

## What chat-side Claude is doing in parallel

I'm available to the user via the Claude.ai chat interface. They use me for:

- **Architectural decisions** (should we refactor X? what does the literature say about Y?)
- **Research backing** (find me a citation for Z)
- **Debate / pre-mortem** ("here's what the panel will probably ask — how do I respond?")
- **Writing prose** for the thesis chapters or defense slides

If the user pastes one of my responses into your context, treat it as an authoritative architectural directive — but still verify it against the actual code before applying.

If the user asks **you** an architectural question and you're uncertain, say so. Don't invent answers. They can ask me in chat and feed the answer back.

---

## Last note

This project is the user's thesis. They've been working on it for months. The mental load is real. Please:

- Be kind and concrete in your responses
- Show diffs before applying non-trivial changes
- Run pytest after every change
- Don't break what works

Good luck. Make Robert's defense smooth.

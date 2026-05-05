# Legacy scripts — preserved for git history

These scripts were part of earlier M.I.N.E.R.V.A. versions but are not
invoked by the v2.8 pipeline. They're preserved here for git history
and reference.

| File | Last canonical version | Replaced by |
|---|---|---|
| `10_prepare_gpt2MINERVA.py` | v2.5 | `10b_prepare_gpt2_neurosymbolic.py` (5-token vs 2-token conditioning) |
| `11_train_gpt2MINERVA.py` | v2.5 | `11b_train_gpt2_neurosymbolic.py` |
| `12_generate_gpt2MINERVA.py` | v2.5 | `12b_generate_gpt2_neurosymbolic.py` |
| `22_build_story_cards_legacy.py` | v2.4 | `30_template_scenario_generator.py` |
| `26_build_release_deck_legacy.py` | v2.4 | `26_faithfulness_audit.py` (note: number reused) |

These files are not unit-tested. Do not import from this directory in
new code. If you need similar functionality, use the canonical replacement.

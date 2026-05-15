#!/usr/bin/env python3
"""Ablation scaffold: GPT-2 generation without control-token conditioning. Run deferred to publication."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("minerva.ablation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MINERVA Script 38: GPT-2 conditioning ablation.")
    p.add_argument("--gpt2_model_dir",
                   default="models/gpt2_tagalog_neurosymbolic",
                   help="Directory with the conditioned GPT-2 fine-tune.")
    p.add_argument("--baseline_corpus",
                   default="data/gpt2_neurosymbolic",
                   help="Directory with train.txt / val.txt / special_tokens.json.")
    p.add_argument("--baseline_promotion",
                   default="reports/merge_gpt2_into_pool.json",
                   help="Existing v2.9.x merge report (used as 'with_conditioning' baseline).")
    p.add_argument("--report_out",
                   default="reports/ablation_no_conditioning.json",
                   help="Where to write the ablation comparison report.")
    p.add_argument("--n_per_label", type=int, default=200,
                   help="How many cards to generate per label for the no-cond pass.")
    p.add_argument("--seed", type=int, default=13,
                   help="Generation seed.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the protocol without executing the no-cond fine-tune + generate.")
    return p.parse_args()


def load_baseline_promotion(path: Path) -> Dict[str, Any]:
    """Read the existing v2.9.x merge report; this becomes the 'with_conditioning' arm."""
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline {path} not found. Run the v2.9 pipeline first to populate it."
        )
    with open(path) as f:
        merge = json.load(f)

    # The merge report has these fields:
    #   gpt2_attempted, gpt2_promoted, gpt2_rejected, by_reject_reason, etc.
    attempted = merge.get("gpt2_attempted") or 0
    promoted = merge.get("gpt2_promoted") or 0
    rate = (100.0 * promoted / attempted) if attempted else 0.0

    return {
        "n_attempted": attempted,
        "n_promoted": promoted,
        "promotion_rate_pct": round(rate, 2),
        "source": str(path),
    }


def describe_protocol() -> str:
    """Return a multi-line description of the no-conditioning ablation protocol."""
    return """
NO-CONDITIONING ABLATION PROTOCOL
==================================

STEP 1: Build a stripped-corpus version
  - Copy data/gpt2_neurosymbolic/{train,val}.txt to a new directory
  - For each line, REMOVE the leading control-token prefix
    (lines that look like: '<|fake|> <|graph_high|> <|qlat_mid|> ... text here')
    become just: 'text here'
  - Save special_tokens.json with an EMPTY token list

STEP 2: Re-fine-tune GPT-2 on the stripped corpus
  - python scripts/11b_train_gpt2_neurosymbolic.py \\
      --corpus_dir data/gpt2_no_conditioning \\
      --base_model jcblaise/gpt2-tagalog \\
      --out_dir models/gpt2_no_conditioning \\
      --epochs 8 --lr 5e-5
  - ~12 min on T4, ~4 min on A100

STEP 3: Generate cards with the stripped model
  - python scripts/12b_generate_gpt2_neurosymbolic.py \\
      --model_dir models/gpt2_no_conditioning \\
      --n 200 --seed 13 \\
      --out generated/gpt2_no_cond_fake_raw.jsonl \\
      --label fake
  - Repeat for label=real

STEP 4: Score and run merge filter (same as the conditioned path)
  - python scripts/13_score_generated_with_qlattice.py ...
  - python scripts/29_merge_gpt2_into_pool.py ...

STEP 5: Compare promotion rates
  - With conditioning: ~30-40% (v2.9.x baseline)
  - Without conditioning: <expected lower>; difference attributable to control tokens

If the delta is < 5 percentage points, the conditioning claim is weaker and
should be reframed in the paper.

EXPECTED RESULT (panel-defense)
================================
H1: Cards from the conditioned model fire more verifiable indicators
    on average than cards from the unconditioned model.
H0: No difference in indicator-firing rate.

Reject H0 with paired-sample t-test on per-card indicator counts (n=200).
"""


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.dry_run:
        logger.info(describe_protocol())
        return 0

    # Real execution path
    baseline_path = Path(args.baseline_promotion)
    logger.info("Loading baseline promotion stats from %s", baseline_path)
    with_conditioning = load_baseline_promotion(baseline_path)
    logger.info("Baseline (with conditioning): %s", with_conditioning)

    # The actual no-conditioning fine-tune + generation is OUT OF SCOPE for
    # this scaffold — see describe_protocol(). We emit a placeholder report
    # that explains how to populate the missing arm.
    report = {
        "version": "v2.9.3",
        "status": "scaffold",
        "with_conditioning": with_conditioning,
        "without_conditioning": {
            "status": "not_run",
            "next_step": (
                "Follow STEP 1-5 in describe_protocol(). Re-run this script "
                "without --dry_run after populating "
                "reports/merge_gpt2_no_conditioning.json"
            ),
        },
        "protocol": describe_protocol(),
    }

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote scaffold report -> %s", out_path)
    logger.info("Run with --dry_run to see the full protocol.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

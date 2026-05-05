#!/usr/bin/env python3
"""
12b_generate_gpt2_neurosymbolic.py  —  v2.6.final

NEURO-SYMBOLIC GPT-2 INFERENCE

Prepends ALL FIVE v2.6.final control tokens to steer GPT-2 toward the
joint distribution P(text | label, graph, qlat, ensem, tier).

This is the inference-time counterpart to scripts/10b and scripts/11b.

WHY THIS MATTERS
----------------
The v2.5 generation script (scripts/12_generate_gpt2MINERVA.py) prepends
two tokens: label + graph. This v2.6.final generator prepends FIVE,
matching the conditioning we baked in at training time. By specifying
high values for all five tokens at inference, we steer the model toward
training examples where:

  - the class label was correct (label),
  - the graph model was confident (graph=high),
  - the symbolic equation was confident (qlat=high),
  - the detector ensemble was confident (ensem=high),
  - the example was teaching-quality clear (tier=novice means clearest).

This means the generator preferentially produces text that, by
construction, will:
  1. Match the requested verdict.
  2. Land in the same neighborhood of the kNN graph as confidently-
     classified training examples.
  3. Have feature signatures that the QLattice equation rates highly.
  4. Be confidently classifiable by the ensemble.
  5. Be clear enough to use as a teaching card.

After generation, we still use the existing detector ensemble accept
gate (script 12 logic) for an independent post-hoc check, AND the new
strict allowlist enforcer (script 33) for personal-name compliance.

USAGE
-----
  # Generate 500 FAKE cards conditioned on high confidence + novice tier
  python scripts/12b_generate_gpt2_neurosymbolic.py \
      --target fake --n 500 \
      --gen_model_dir models/gpt2_tagalog_neurosymbolic \
      --graph high --qlat high --ensem high --tier novice \
      --out generated/gpt2_neuro_fake.jsonl

  # Generate 500 REAL cards with mid-confidence + advanced (harder cases)
  python scripts/12b_generate_gpt2_neurosymbolic.py \
      --target real --n 500 \
      --graph mid --qlat mid --ensem mid --tier advanced \
      --out generated/gpt2_neuro_real_advanced.jsonl

CITATIONS
---------
- Keskar et al. (2019). CTRL: Conditional Transformer Language Model.
- Dathathri et al. (2020). Plug and Play Language Models. ICLR.
- Cruz, Tan, & Cheng (2020). LREC. — base model jcblaise/gpt2-tagalog.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Must match scripts/10b_prepare_gpt2_neurosymbolic.py
LABEL_TOK = {"real": "<|label=real|>", "fake": "<|label=fake|>"}
GRAPH_TOK = {"high": "<|graph=high|>", "mid": "<|graph=mid|>",
             "low": "<|graph=low|>", "unk": "<|graph=unk|>"}
QLAT_TOK = {"high": "<|qlat=high|>", "mid": "<|qlat=mid|>",
            "low": "<|qlat=low|>", "unk": "<|qlat=unk|>"}
ENSEM_TOK = {"high": "<|ensem=high|>", "mid": "<|ensem=mid|>",
             "low": "<|ensem=low|>", "unk": "<|ensem=unk|>"}
TIER_TOK = {"novice": "<|tier=novice|>", "proficient": "<|tier=proficient|>",
            "advanced": "<|tier=advanced|>", "unk": "<|tier=unk|>"}


def main() -> None:
    p = argparse.ArgumentParser(
        description="v2.6.final neuro-symbolic GPT-2 inference."
    )
    p.add_argument("--target", choices=["real", "fake"], required=True)
    p.add_argument("--n", type=int, required=True,
                   help="Number of generations to produce")
    p.add_argument("--gen_model_dir",
                   default="models/gpt2_tagalog_neurosymbolic")
    p.add_argument("--out", required=True,
                   help="Output JSONL (one generation per line)")

    # Conditioning tokens
    p.add_argument("--graph", default="high",
                   choices=["high", "mid", "low", "unk"])
    p.add_argument("--qlat",  default="high",
                   choices=["high", "mid", "low", "unk"])
    p.add_argument("--ensem", default="high",
                   choices=["high", "mid", "low", "unk"])
    p.add_argument("--tier",  default="novice",
                   choices=["novice", "proficient", "advanced", "unk"])

    # Generation hyperparameters
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=13)

    p.add_argument("--report_out",
                   default="reports/gpt2_neurosymbolic_generation.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, set_seed
        )
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}\n"
            "Install with: pip install transformers torch"
        )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load
    logger.info("Loading neuro-symbolic GPT-2 from %s", args.gen_model_dir)
    tok = AutoTokenizer.from_pretrained(args.gen_model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only models prefer left padding
    model = AutoModelForCausalLM.from_pretrained(args.gen_model_dir).to(device)
    model.eval()

    # 2. Build the conditioning prompt
    prompt = " ".join([
        LABEL_TOK[args.target],
        GRAPH_TOK[args.graph],
        QLAT_TOK[args.qlat],
        ENSEM_TOK[args.ensem],
        TIER_TOK[args.tier],
    ]) + " "
    logger.info("Conditioning prompt: %s", prompt)

    # 3. Generate
    logger.info("Generating %d samples for target=%s, graph=%s, qlat=%s, "
                "ensem=%s, tier=%s",
                args.n, args.target, args.graph, args.qlat, args.ensem, args.tier)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with out_path.open("w", encoding="utf-8") as f:
        while n_done < args.n:
            batch_n = min(args.batch_size, args.n - n_done)
            inputs = tok([prompt] * batch_n, return_tensors="pt",
                         padding=True).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tok.pad_token_id,
                )
            # Strip the prompt tokens from each generation
            for ids, in_ids in zip(out_ids, inputs["input_ids"]):
                gen_only = ids[in_ids.shape[0]:]
                text = tok.decode(gen_only, skip_special_tokens=True).strip()
                row = {
                    "target": args.target,
                    "control_tokens": {
                        "label": args.target,
                        "graph": args.graph,
                        "qlat":  args.qlat,
                        "ensem": args.ensem,
                        "tier":  args.tier,
                    },
                    "text": text,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1
            logger.info("  ... %d / %d", n_done, args.n)

    logger.info("Wrote %d generations -> %s", n_done, out_path)

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "version": "v2.6.final",
        "gen_model_dir": args.gen_model_dir,
        "n_generated": n_done,
        "conditioning": {
            "label": args.target,
            "graph": args.graph,
            "qlat":  args.qlat,
            "ensem": args.ensem,
            "tier":  args.tier,
        },
        "hyperparameters": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "out_jsonl": str(out_path),
        "next_steps": [
            "Run scripts/13_score_generated_with_qlattice.py to score outputs",
            "Run scripts/33_strict_name_allowlist.py to enforce candidate-only naming",
            "Run scripts/24_curate_teaching_cards.py to build the final pool",
        ],
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("Neuro-symbolic generation complete (v2.6.final)")
    logger.info("  Conditioning: label=%s graph=%s qlat=%s ensem=%s tier=%s",
                args.target, args.graph, args.qlat, args.ensem, args.tier)
    logger.info("  Output -> %s", out_path)
    logger.info("  Report -> %s", args.report_out)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

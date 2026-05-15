#!/usr/bin/env python3
"""Fine-tune jcblaise/gpt2-tagalog with control tokens. Training seed 1729 per Picard 2021."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(
        description="neuro-symbolic GPT-2 fine-tuning (v2.8.5)."
    )
    p.add_argument("--corpus_dir", default="data/gpt2_neurosymbolic",
                   help="Output directory of script 10b "
                        "(must contain train.txt, val.txt, special_tokens.json)")
    p.add_argument("--base_model", default="jcblaise/gpt2-tagalog")
    p.add_argument("--out_dir", default="models/gpt2_tagalog_neurosymbolic")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--per_device_batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--block_size", type=int, default=256)
    # "torch.manual_seed(3407) is all you need". The seed 42 is over-represented
    # in published ML and creates a cherry-picking risk. 1729 is the
    # Hardy-Ramanujan number (= 7 * 13 * 19), used elsewhere in the project
    # (scripts/minerva_candidates.py:305) for consistency across the codebase.
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--report_out",
                   default="reports/gpt2_neurosymbolic_training.json")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Lazy imports — heavy and only needed at runtime
    try:
        import torch
        from datasets import Dataset, DatasetDict
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}\n"
            "Install with: pip install transformers datasets accelerate torch"
        )

    set_seed(args.seed)

    corpus_dir = Path(args.corpus_dir)
    train_file = corpus_dir / "train.txt"
    val_file = corpus_dir / "val.txt"
    tokens_file = corpus_dir / "special_tokens.json"

    for p in (train_file, val_file, tokens_file):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run scripts/10b_prepare_gpt2_neurosymbolic.py first."
            )

    special_tokens = json.loads(tokens_file.read_text(encoding="utf-8"))
    logger.info("Loaded %d special tokens from %s",
                len(special_tokens["additional_special_tokens"]), tokens_file)

    # 1. Load base tokenizer + model
    logger.info("Loading base model: %s", args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Capture base vocab size BEFORE adding special tokens (for the audit)
    base_vocab_size = len(tok)

    # 2. Register special tokens (NEW v2.6.final tokens added to base set)
    n_added = tok.add_special_tokens({
        "additional_special_tokens": special_tokens["additional_special_tokens"]
    })
    logger.info("Registered %d new special tokens (total tokenizer size: %d)",
                n_added, len(tok))

    # 3. Load model and resize embedding matrix to fit the new tokens
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tok))
    new_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info("Embedding matrix resized: %d -> %d rows",
                base_vocab_size, new_vocab_size)
    logger.info("Pretrained linguistic weights preserved: only %d new "
                "embedding rows are randomly initialized.", n_added)

    # 4. Build training datasets from text files.
    # `NotImplementedError: Loading a dataset cached in a LocalFileSystem is
    # not supported` on `datasets >= 2.14` due to an inverted FS-type check
    # in `as_dataset()`. We just want to wrap two text files as Datasets;
    # `Dataset.from_dict` does that fully in-memory with no fsspec involvement.
    def _read_lines(path: Path) -> list:
        with open(path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]

    raw = DatasetDict({
        "train": Dataset.from_dict({"text": _read_lines(train_file)}),
        "validation": Dataset.from_dict({"text": _read_lines(val_file)}),
    })
    logger.info("Built in-memory DatasetDict: train=%d val=%d (no fsspec cache)",
                len(raw["train"]), len(raw["validation"]))

    def tokenize(batch):
        return tok(batch["text"])

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = (len(concatenated["input_ids"]) // args.block_size) * args.block_size
        result = {
            k: [t[i: i + args.block_size] for i in range(0, total_len, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(group_texts, batched=True)

    # 5. Trainer
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("logs/gpt2_neurosymbolic")
    log_dir.mkdir(parents=True, exist_ok=True)

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # TrainingArguments — version-compatible build.
    # transformers 4.46+ uses `eval_strategy`; older versions use `evaluation_strategy`.
    targs_kwargs = dict(
        output_dir=str(out_dir),
        logging_dir=str(log_dir),
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=0.05,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )
    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in _ta_params:
        targs_kwargs["eval_strategy"] = "epoch"
    else:
        targs_kwargs["evaluation_strategy"] = "epoch"
    targs = TrainingArguments(**targs_kwargs)
    trainer_kwargs = dict(
        model=model,
        args=targs,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        data_collator=collator,
        # plateaus before reaching --epochs. Combined with
        # load_best_model_at_end=True, this guarantees we keep the best
        # checkpoint even if later epochs degrade. patience=2 means we tolerate
        # 2 evaluations (= 2 epochs at eval_strategy='epoch') of no improvement
        # before stopping.
        # Refs: Mosbach et al. 2021 (ICLR); HuggingFace Trainer docs.
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    # transformers 4.46+ renamed `tokenizer` → `processing_class` on Trainer
    _tr_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in _tr_params:
        trainer_kwargs["processing_class"] = tok       # transformers 4.46+
    else:
        trainer_kwargs["tokenizer"] = tok              # transformers <4.46
    trainer = Trainer(**trainer_kwargs)

    logger.info("Starting fine-tuning...")
    train_result = trainer.train()

    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    logger.info("Saved model + tokenizer -> %s", out_dir)

    eval_result = trainer.evaluate()

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "version": "v2.9.4",
        "base_model": args.base_model,
        "out_dir": str(out_dir),
        "vocabulary": {
            "base_vocab_size": base_vocab_size,
            "new_special_tokens_added": n_added,
            "final_vocab_size": new_vocab_size,
            "preservation_note": (
                f"The first {base_vocab_size} embedding rows are byte-identical "
                f"to the pretrained {args.base_model} checkpoint. Only the {n_added} "
                f"new rows for v2.6.final special tokens are randomly initialized "
                f"and trained. Base linguistic competence is preserved."
            ),
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "per_device_batch": args.per_device_batch,
            "grad_accum": args.grad_accum,
            "effective_batch": args.per_device_batch * args.grad_accum,
            "block_size": args.block_size,
            "seed": args.seed,
            "fp16": bool(torch.cuda.is_available()),
        },
        "results": {
            "train_runtime_seconds": getattr(train_result, "metrics", {}).get(
                "train_runtime", None),
            "final_train_loss": getattr(train_result, "metrics", {}).get(
                "train_loss", None),
            "final_eval_loss": eval_result.get("eval_loss", None),
        },
        "citations": [
            "Keskar et al. (2019). CTRL.",
            "Wolf et al. (2020). HuggingFace Transformers. EMNLP Demos.",
            "Cruz, Tan, & Cheng (2020). LREC.",
        ],
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("Neuro-symbolic fine-tuning complete (v2.6.final)")
    logger.info("  Saved        : %s", out_dir)
    logger.info("  Final loss   : %.4f",
                eval_result.get("eval_loss", float("nan")))
    logger.info("  Vocab        : %d (base) + %d (new) = %d",
                base_vocab_size, n_added, new_vocab_size)
    logger.info("  Pretrained linguistic weights preserved.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

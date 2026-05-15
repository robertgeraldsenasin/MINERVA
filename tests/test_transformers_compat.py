"""Unit tests for the transformers API compat shim used by scripts 16 and 11b.

The shim must select the correct kwarg name based on inspect.signature:
  - transformers 4.46+: TrainingArguments(eval_strategy=...), Trainer(processing_class=...)
  - transformers <4.46: TrainingArguments(evaluation_strategy=...), Trainer(tokenizer=...)

These tests verify the inspect-based logic against fake "old" and "new" API
classes. They run hermetically — no real transformers installation is exercised.

Run:
    python -m pytest tests/test_transformers_compat.py -v
"""

from __future__ import annotations

import inspect


# Fake "old API" (transformers <4.46) — uses evaluation_strategy + tokenizer

class _OldTrainingArguments:
    def __init__(self, output_dir=None, evaluation_strategy=None,
                 save_strategy=None, **kwargs):
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.kwargs = kwargs


class _OldTrainer:
    def __init__(self, model=None, args=None, train_dataset=None,
                 eval_dataset=None, tokenizer=None, data_collator=None,
                 compute_metrics=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics


# Fake "new API" (transformers 4.46+) — uses eval_strategy + processing_class

class _NewTrainingArguments:
    def __init__(self, output_dir=None, eval_strategy=None,
                 save_strategy=None, **kwargs):
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.save_strategy = save_strategy
        self.kwargs = kwargs


class _NewTrainer:
    def __init__(self, model=None, args=None, train_dataset=None,
                 eval_dataset=None, processing_class=None, data_collator=None,
                 compute_metrics=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics


# The shim being tested — same logic as in scripts/16 and scripts/11b

def build_training_args(TrainingArgumentsCls, **kwargs):
    """Replicate the shim used in scripts 16 and 11b."""
    params = inspect.signature(TrainingArgumentsCls.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("_eval_value", "epoch")
    else:
        kwargs["evaluation_strategy"] = kwargs.pop("_eval_value", "epoch")
    return TrainingArgumentsCls(**kwargs)


def build_trainer(TrainerCls, tokenizer_obj, **kwargs):
    """Replicate the Trainer shim used in scripts 16 and 11b."""
    params = inspect.signature(TrainerCls.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer_obj
    else:
        kwargs["tokenizer"] = tokenizer_obj
    return TrainerCls(**kwargs)


# Tests against the OLD API

class TestOldTransformersAPI:
    def test_uses_evaluation_strategy_on_old(self):
        ta = build_training_args(
            _OldTrainingArguments,
            output_dir="/tmp/x",
            save_strategy="epoch",
        )
        assert ta.evaluation_strategy == "epoch"

    def test_uses_tokenizer_kwarg_on_old(self):
        sentinel_tok = object()
        tr = build_trainer(
            _OldTrainer,
            sentinel_tok,
            model=None, args=None,
            train_dataset=None, eval_dataset=None,
            data_collator=None, compute_metrics=None,
        )
        assert tr.tokenizer is sentinel_tok

    def test_old_api_does_not_choke_on_eval_strategy_kwarg(self):
        """If we accidentally passed eval_strategy to old API, it would TypeError.
        Verify the shim never does that."""
        # The shim should only inject evaluation_strategy, not eval_strategy.
        # We confirm by passing through and checking no TypeError is raised.
        ta = build_training_args(_OldTrainingArguments, output_dir="/tmp")
        # Build success = no TypeError. The exact attribute matters less than
        # not throwing.
        assert ta.evaluation_strategy == "epoch"


# Tests against the NEW API

class TestNewTransformersAPI:
    def test_uses_eval_strategy_on_new(self):
        ta = build_training_args(
            _NewTrainingArguments,
            output_dir="/tmp/x",
            save_strategy="epoch",
        )
        assert ta.eval_strategy == "epoch"

    def test_uses_processing_class_on_new(self):
        sentinel_tok = object()
        tr = build_trainer(
            _NewTrainer,
            sentinel_tok,
            model=None, args=None,
            train_dataset=None, eval_dataset=None,
            data_collator=None, compute_metrics=None,
        )
        assert tr.processing_class is sentinel_tok

    def test_new_api_does_not_receive_old_kwargs(self):
        """If shim accidentally passed evaluation_strategy to new API,
        it would TypeError because new API doesn't accept that name."""
        ta = build_training_args(_NewTrainingArguments, output_dir="/tmp")
        assert ta.eval_strategy == "epoch"
        # And the old attribute should NOT be present (we never set it)
        assert not hasattr(ta, "evaluation_strategy") or ta.evaluation_strategy is None


# Sanity: real script files contain the shim

class TestShimPresentInScripts:
    """Verify the actual scripts have the shim, in case a future edit removes it."""

    def test_script_16_uses_eval_strategy_shim(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1]
               / "scripts" / "16_train_transformer_classifier.py").read_text()
        assert "inspect.signature(TrainingArguments.__init__).parameters" in src
        assert '"eval_strategy"' in src
        assert '"evaluation_strategy"' in src
        assert "inspect.signature(Trainer.__init__).parameters" in src
        assert '"processing_class"' in src

    def test_script_11b_uses_eval_strategy_shim(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parents[1]
               / "scripts" / "11b_train_gpt2_neurosymbolic.py").read_text()
        assert "inspect.signature(TrainingArguments.__init__).parameters" in src
        assert '"eval_strategy"' in src
        assert '"evaluation_strategy"' in src
        assert "inspect.signature(Trainer.__init__).parameters" in src
        assert '"processing_class"' in src

"""
M.I.N.E.R.V.A. v2.9.3 — Centralized hyperparameter configuration

This module is the SINGLE SOURCE OF TRUTH for all training hyperparameters used
across the pipeline. Both the notebook and the per-script CLIs default to these
values; importing from here ensures no drift between what the notebook does and
what the scripts do.

Citations supporting each value are inline below — these are panel-defense ammo.

Override at runtime by setting environment variables (handy for quick experiments
without code changes):

    MINERVA_TRAIN_SEEDS="13,29,47,89,127"      # comma-separated seeds
    MINERVA_TRAIN_EPOCHS=3                     # detector epochs
    MINERVA_GPT2_EPOCHS=8                      # GPT-2 fine-tune epochs
    MINERVA_GPT2_TRAIN_BATCH=8                 # GPT-2 per-device batch (A100)
    MINERVA_DETECTOR_BATCH=8                   # detector per-device batch
    MINERVA_GRAD_ACCUM=4                       # gradient accumulation steps
    MINERVA_DETECTOR_LR=2e-5                   # detector learning rate
    MINERVA_GPT2_LR=5e-5                       # GPT-2 learning rate

Usage in a script:
    from minerva_config import CONFIG
    print(CONFIG.train_seeds)         # → [13, 29, 47, 89, 127]
    print(CONFIG.detector_batch)      # → 8
    print(CONFIG.effective_batch)     # → 32 (8 × 4 grad_accum)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


# ============================================================================
# DEFAULT VALUES (override via env vars; see module docstring)
# ============================================================================

# ---- Random seeds for detector training (RoBERTa + DistilBERT × 5 seeds) ----
# Standard: 5 seeds (Liu et al. 2019 RoBERTa: "median over five runs").
# Prime numbers chosen to avoid the 42 / small-integer cherry-picking risk
# (Picard 2021, "torch.manual_seed(3407) is all you need").
#
# References:
#   Liu, Y. et al. (2019). RoBERTa. arXiv:1907.11692
#   Dodge, J. et al. (2020). Fine-Tuning Pretrained Language Models. arXiv:2002.06305
#   Mosbach, M. et al. (2021). On the Stability of Fine-tuning BERT. ICLR.
#   Picard, D. (2021). torch.manual_seed(3407) is all you need. arXiv:2109.08203
DEFAULT_TRAIN_SEEDS = [13, 29, 47, 89, 127]


# ---- Detector (RoBERTa-Tagalog + DistilBERT-multilingual) ----
# Devlin et al. (2019) BERT recommends:
#   - Batch size: 16 or 32
#   - Learning rate: 2e-5, 3e-5, 5e-5
#   - Epochs: 2, 3, 4
# Liu et al. (2019) RoBERTa Appendix C, Table 10:
#   - Batch size: {16, 32}
#   - Learning rate: {1e-5, 2e-5, 3e-5}
#   - Epochs: 10 with early stopping (smaller-data) or 3 (larger-data)
DEFAULT_DETECTOR_BATCH = 8     # per-device; ×4 grad_accum = effective 32 (Devlin std)
DEFAULT_GRAD_ACCUM     = 4
DEFAULT_DETECTOR_LR    = 2e-5  # Liu 2019 Table 10 most common
DEFAULT_DETECTOR_EPOCHS = 3    # Devlin 2019 + early stopping in v2.9.3
DEFAULT_DETECTOR_MAX_LEN = 256
DEFAULT_DETECTOR_WARMUP_RATIO = 0.10
DEFAULT_DETECTOR_WEIGHT_DECAY = 0.01

# Early stopping (added v2.9.3 per Mosbach 2021 + HF best practice)
DEFAULT_DETECTOR_PATIENCE = 1  # detector usually converges by epoch 2


# ---- GPT-2 Tagalog (control-token conditional fine-tune) ----
# v2.8.6 evidence: 3 epochs only converged to eval_loss=3.49 (insufficient).
# v2.8.7 ran 8 epochs and reached eval_loss=3.29 — better but still
# decreasing. v2.9.3 keeps 8 as upper bound but adds early stopping
# so we automatically stop if loss plateaus before epoch 8.
#
# Howard & Ruder 2018 ULMFiT: "fine-tune for 4-10 epochs depending on
# task and dataset size" — JCBlaise post-pseudonymize is small enough
# that 8 is the right ballpark.
DEFAULT_GPT2_EPOCHS    = 8       # upper bound; early stopping cuts short if plateaued
DEFAULT_GPT2_LR        = 5e-5    # standard for GPT-2 fine-tuning
DEFAULT_GPT2_TRAIN_BATCH = 2     # per-device on T4 (15GB VRAM); auto-bumped on A100
DEFAULT_GPT2_PATIENCE  = 2       # 2 evaluations without improvement → stop


# ---- GPT-2 generation (post-fine-tune) ----
# This is the persistent-generation loop's per-attempt pool size, NOT a
# training batch (renamed v2.9.3 from the misleading GPT2_BATCH_SIZE).
# Tuned so that ~10-30% promotion rate yields ~50-150 cards per attempt.
DEFAULT_GPT2_GEN_POOL_SIZE       = 500
DEFAULT_GPT2_INFERENCE_BATCH     = 16   # actual model.generate() batch
DEFAULT_GPT2_MIN_PROMOTED_PER_LABEL = 100
DEFAULT_GPT2_MAX_ATTEMPTS        = 4
DEFAULT_GPT2_MAX_NEW_TOKENS      = 200


# ---- Pipeline-wide RNG seed (template generation, deck draw, etc.) ----
# 1729 = Hardy-Ramanujan number; deliberately distinct from the 42 default.
DEFAULT_RNG_SEED = 1729


# ============================================================================
# CONFIG LOADER (env-var aware)
# ============================================================================

def _env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_seed_list(key: str, default: List[int]) -> List[int]:
    val = os.environ.get(key)
    if not val:
        return list(default)
    try:
        seeds = [int(s.strip()) for s in val.split(",") if s.strip()]
        return seeds or list(default)
    except ValueError:
        return list(default)


@dataclass(frozen=True)
class MinervaConfig:
    """Resolved hyperparameter configuration. Immutable per-process."""

    # Seeds
    train_seeds: List[int] = field(default_factory=list)
    rng_seed: int = DEFAULT_RNG_SEED

    # Detector training
    detector_batch: int = DEFAULT_DETECTOR_BATCH
    grad_accum: int = DEFAULT_GRAD_ACCUM
    detector_lr: float = DEFAULT_DETECTOR_LR
    detector_epochs: int = DEFAULT_DETECTOR_EPOCHS
    detector_max_len: int = DEFAULT_DETECTOR_MAX_LEN
    detector_warmup_ratio: float = DEFAULT_DETECTOR_WARMUP_RATIO
    detector_weight_decay: float = DEFAULT_DETECTOR_WEIGHT_DECAY
    detector_patience: int = DEFAULT_DETECTOR_PATIENCE

    # GPT-2
    gpt2_epochs: int = DEFAULT_GPT2_EPOCHS
    gpt2_lr: float = DEFAULT_GPT2_LR
    gpt2_train_batch: int = DEFAULT_GPT2_TRAIN_BATCH
    gpt2_patience: int = DEFAULT_GPT2_PATIENCE

    # GPT-2 generation
    gpt2_gen_pool_size: int = DEFAULT_GPT2_GEN_POOL_SIZE
    gpt2_inference_batch: int = DEFAULT_GPT2_INFERENCE_BATCH
    gpt2_min_promoted_per_label: int = DEFAULT_GPT2_MIN_PROMOTED_PER_LABEL
    gpt2_max_attempts: int = DEFAULT_GPT2_MAX_ATTEMPTS
    gpt2_max_new_tokens: int = DEFAULT_GPT2_MAX_NEW_TOKENS

    @property
    def effective_batch(self) -> int:
        """Effective training batch after gradient accumulation (Devlin 2019: 32 is std)."""
        return self.detector_batch * self.grad_accum

    @property
    def n_seeds(self) -> int:
        return len(self.train_seeds)

    def to_dict(self) -> dict:
        """Serializable form for logging into reports."""
        return {
            "train_seeds": list(self.train_seeds),
            "n_seeds": self.n_seeds,
            "rng_seed": self.rng_seed,
            "detector": {
                "batch": self.detector_batch,
                "grad_accum": self.grad_accum,
                "effective_batch": self.effective_batch,
                "lr": self.detector_lr,
                "epochs": self.detector_epochs,
                "max_len": self.detector_max_len,
                "warmup_ratio": self.detector_warmup_ratio,
                "weight_decay": self.detector_weight_decay,
                "early_stopping_patience": self.detector_patience,
            },
            "gpt2": {
                "epochs": self.gpt2_epochs,
                "lr": self.gpt2_lr,
                "train_batch": self.gpt2_train_batch,
                "early_stopping_patience": self.gpt2_patience,
                "gen_pool_size": self.gpt2_gen_pool_size,
                "inference_batch": self.gpt2_inference_batch,
                "min_promoted_per_label": self.gpt2_min_promoted_per_label,
                "max_attempts": self.gpt2_max_attempts,
                "max_new_tokens": self.gpt2_max_new_tokens,
            },
            "config_version": "v2.9.3",
        }


def load_config() -> MinervaConfig:
    """Load config from environment variables, falling back to defaults."""
    return MinervaConfig(
        train_seeds=_env_seed_list("MINERVA_TRAIN_SEEDS", DEFAULT_TRAIN_SEEDS),
        rng_seed=_env_int("MINERVA_RNG_SEED", DEFAULT_RNG_SEED),

        detector_batch=_env_int("MINERVA_DETECTOR_BATCH", DEFAULT_DETECTOR_BATCH),
        grad_accum=_env_int("MINERVA_GRAD_ACCUM", DEFAULT_GRAD_ACCUM),
        detector_lr=_env_float("MINERVA_DETECTOR_LR", DEFAULT_DETECTOR_LR),
        detector_epochs=_env_int("MINERVA_TRAIN_EPOCHS", DEFAULT_DETECTOR_EPOCHS),
        detector_max_len=_env_int("MINERVA_DETECTOR_MAX_LEN", DEFAULT_DETECTOR_MAX_LEN),
        detector_warmup_ratio=_env_float("MINERVA_DETECTOR_WARMUP", DEFAULT_DETECTOR_WARMUP_RATIO),
        detector_weight_decay=_env_float("MINERVA_DETECTOR_WD", DEFAULT_DETECTOR_WEIGHT_DECAY),
        detector_patience=_env_int("MINERVA_DETECTOR_PATIENCE", DEFAULT_DETECTOR_PATIENCE),

        gpt2_epochs=_env_int("MINERVA_GPT2_EPOCHS", DEFAULT_GPT2_EPOCHS),
        gpt2_lr=_env_float("MINERVA_GPT2_LR", DEFAULT_GPT2_LR),
        gpt2_train_batch=_env_int("MINERVA_GPT2_TRAIN_BATCH", DEFAULT_GPT2_TRAIN_BATCH),
        gpt2_patience=_env_int("MINERVA_GPT2_PATIENCE", DEFAULT_GPT2_PATIENCE),

        gpt2_gen_pool_size=_env_int("MINERVA_GPT2_GEN_POOL", DEFAULT_GPT2_GEN_POOL_SIZE),
        gpt2_inference_batch=_env_int("MINERVA_GPT2_INFER_BATCH", DEFAULT_GPT2_INFERENCE_BATCH),
        gpt2_min_promoted_per_label=_env_int("MINERVA_GPT2_MIN_PROMOTED", DEFAULT_GPT2_MIN_PROMOTED_PER_LABEL),
        gpt2_max_attempts=_env_int("MINERVA_GPT2_MAX_ATTEMPTS", DEFAULT_GPT2_MAX_ATTEMPTS),
        gpt2_max_new_tokens=_env_int("MINERVA_GPT2_MAX_NEW_TOKENS", DEFAULT_GPT2_MAX_NEW_TOKENS),
    )


# Module-level singleton — import this anywhere as `from minerva_config import CONFIG`
CONFIG: MinervaConfig = load_config()


# ============================================================================
# Hardware-aware adjustment helpers
# ============================================================================

def gpt2_train_batch_for_gpu(gpu_name: str | None = None) -> int:
    """Return the right GPT-2 per-device batch given GPU memory.

    A100 (40 GB)  → 8
    A100 (80 GB)  → 16
    V100 (32 GB)  → 4
    T4 (15 GB)    → 2 (default)
    """
    if gpu_name is None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            gpu_name = ""

    name = (gpu_name or "").upper()
    if "A100" in name and "80" in name:
        return 16
    if "A100" in name:
        return 8
    if "V100" in name:
        return 4
    return DEFAULT_GPT2_TRAIN_BATCH  # 2


def detector_batch_for_gpu(gpu_name: str | None = None) -> tuple[int, int]:
    """Return (per-device batch, grad_accum) maintaining effective batch=32.

    A100 (40 GB)  → (32, 1)
    V100 (32 GB)  → (16, 2)
    T4 (15 GB)    → (8, 4)  ← default
    """
    if gpu_name is None:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            gpu_name = ""

    name = (gpu_name or "").upper()
    if "A100" in name:
        return (32, 1)
    if "V100" in name:
        return (16, 2)
    return (DEFAULT_DETECTOR_BATCH, DEFAULT_GRAD_ACCUM)


__all__ = [
    "CONFIG", "MinervaConfig", "load_config",
    "gpt2_train_batch_for_gpu", "detector_batch_for_gpu",
    "DEFAULT_TRAIN_SEEDS", "DEFAULT_DETECTOR_EPOCHS",
    "DEFAULT_GPT2_EPOCHS", "DEFAULT_DETECTOR_BATCH",
    "DEFAULT_GRAD_ACCUM", "DEFAULT_GPT2_GEN_POOL_SIZE",
]

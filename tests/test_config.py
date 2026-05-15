"""Unit tests for v2.9.3 scripts/minerva_config.py.

Verifies:
  - Default values match published-paper protocols (Devlin 2019, Liu 2019, Mosbach 2021)
  - Environment variables override defaults correctly
  - Hardware-aware GPU batch sizing returns sensible values
  - effective_batch property computes correctly

Run:
    python -m pytest tests/test_config.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable
REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# Import once, lazily — env vars apply on import only for module-level CONFIG
import minerva_config


# Default values match published-paper protocols

class TestDefaultsMatchPublishedProtocols:
    """The defaults are the v2.9.3 audit recommendations.

    Don't change these without updating the citations in the docstring.
    """

    def test_5_seeds_minimum(self):
        """Liu 2019 RoBERTa: 'median over five runs'. Need >=5 seeds."""
        assert len(minerva_config.DEFAULT_TRAIN_SEEDS) >= 5

    def test_seeds_avoid_42_cherry_picking(self):
        """Picard 2021: 'random seed tuning could be regarded as cherry-picking'.
        Avoid 42 in default seed set."""
        assert 42 not in minerva_config.DEFAULT_TRAIN_SEEDS

    def test_seeds_are_distinct(self):
        """No duplicates in the seed list."""
        assert len(set(minerva_config.DEFAULT_TRAIN_SEEDS)) == len(minerva_config.DEFAULT_TRAIN_SEEDS)

    def test_detector_batch_matches_devlin(self):
        """Devlin 2019: batch 16 or 32. With grad_accum=4, batch=8 -> effective 32."""
        b = minerva_config.DEFAULT_DETECTOR_BATCH
        ga = minerva_config.DEFAULT_GRAD_ACCUM
        effective = b * ga
        assert effective in (16, 32)

    def test_detector_lr_in_liu_range(self):
        """Liu 2019 Table 10: {1e-5, 2e-5, 3e-5}. Allow up to 5e-5."""
        assert 1e-5 <= minerva_config.DEFAULT_DETECTOR_LR <= 5e-5

    def test_detector_epochs_in_devlin_range(self):
        """Devlin 2019: 2, 3, or 4 epochs (with early stopping in v2.9.3)."""
        assert 2 <= minerva_config.DEFAULT_DETECTOR_EPOCHS <= 10

    def test_gpt2_epochs_reasonable_for_small_corpus(self):
        """Howard & Ruder 2018 ULMFiT: 4-10 epochs for small-corpus LM fine-tune."""
        assert 4 <= minerva_config.DEFAULT_GPT2_EPOCHS <= 10


# Environment-variable loading

class TestEnvVarLoading:
    def test_seeds_override(self, monkeypatch):
        monkeypatch.setenv("MINERVA_TRAIN_SEEDS", "1,2,3,4,5,6,7")
        cfg = minerva_config.load_config()
        assert cfg.train_seeds == [1, 2, 3, 4, 5, 6, 7]
        assert cfg.n_seeds == 7

    def test_detector_batch_override(self, monkeypatch):
        monkeypatch.setenv("MINERVA_DETECTOR_BATCH", "16")
        monkeypatch.setenv("MINERVA_GRAD_ACCUM", "2")
        cfg = minerva_config.load_config()
        assert cfg.detector_batch == 16
        assert cfg.grad_accum == 2
        assert cfg.effective_batch == 32

    def test_gpt2_epochs_override(self, monkeypatch):
        monkeypatch.setenv("MINERVA_GPT2_EPOCHS", "12")
        cfg = minerva_config.load_config()
        assert cfg.gpt2_epochs == 12

    def test_invalid_int_falls_back_to_default(self, monkeypatch):
        """Defensive: bad env-var values should not crash, just use defaults."""
        monkeypatch.setenv("MINERVA_DETECTOR_BATCH", "not_a_number")
        cfg = minerva_config.load_config()
        assert cfg.detector_batch == minerva_config.DEFAULT_DETECTOR_BATCH

    def test_empty_seed_list_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MINERVA_TRAIN_SEEDS", "")
        cfg = minerva_config.load_config()
        assert cfg.train_seeds == minerva_config.DEFAULT_TRAIN_SEEDS


# effective_batch property

class TestEffectiveBatch:
    def test_8x4_equals_32(self):
        cfg = minerva_config.MinervaConfig(detector_batch=8, grad_accum=4)
        assert cfg.effective_batch == 32

    def test_16x2_equals_32(self):
        cfg = minerva_config.MinervaConfig(detector_batch=16, grad_accum=2)
        assert cfg.effective_batch == 32

    def test_32x1_equals_32(self):
        cfg = minerva_config.MinervaConfig(detector_batch=32, grad_accum=1)
        assert cfg.effective_batch == 32


# Hardware-aware sizing

class TestGPUSizing:
    def test_a100_40gb_gpt2_batch(self):
        assert minerva_config.gpt2_train_batch_for_gpu("NVIDIA A100-SXM4-40GB") == 8

    def test_a100_80gb_gpt2_batch(self):
        assert minerva_config.gpt2_train_batch_for_gpu("NVIDIA A100 80GB") == 16

    def test_t4_gpt2_batch(self):
        assert minerva_config.gpt2_train_batch_for_gpu("Tesla T4") == 2

    def test_v100_gpt2_batch(self):
        assert minerva_config.gpt2_train_batch_for_gpu("Tesla V100") == 4

    def test_unknown_gpu_falls_back_to_t4_default(self):
        assert minerva_config.gpt2_train_batch_for_gpu("Unknown GPU") == 2

    def test_a100_detector_batch_is_32_no_accum(self):
        """On A100 (enough VRAM) we run native batch=32, no grad accum needed."""
        b, ga = minerva_config.detector_batch_for_gpu("NVIDIA A100-SXM4-40GB")
        assert b * ga == 32
        assert ga == 1  # no accumulation needed

    def test_t4_detector_batch_uses_grad_accum(self):
        """T4 needs grad_accum to reach effective 32."""
        b, ga = minerva_config.detector_batch_for_gpu("Tesla T4")
        assert b * ga == 32
        assert ga > 1  # accumulation needed


# Serialization for reports

class TestSerialization:
    def test_to_dict_has_expected_keys(self):
        cfg = minerva_config.MinervaConfig(train_seeds=[13, 29, 47, 89, 127])
        d = cfg.to_dict()
        assert "train_seeds" in d
        assert "n_seeds" in d
        assert "detector" in d
        assert "gpt2" in d
        assert "config_version" in d
        assert d["config_version"] == "v2.9.3"

    def test_to_dict_includes_effective_batch(self):
        cfg = minerva_config.MinervaConfig(detector_batch=8, grad_accum=4)
        d = cfg.to_dict()
        assert d["detector"]["effective_batch"] == 32

    def test_to_dict_is_json_serializable(self):
        import json
        cfg = minerva_config.MinervaConfig(train_seeds=[1, 2, 3, 4, 5])
        d = cfg.to_dict()
        s = json.dumps(d)  # should not raise
        assert "train_seeds" in s


# CONFIG singleton present

class TestConfigSingleton:
    def test_config_is_minerva_config_instance(self):
        assert isinstance(minerva_config.CONFIG, minerva_config.MinervaConfig)

    def test_config_has_seeds(self):
        assert len(minerva_config.CONFIG.train_seeds) >= 5

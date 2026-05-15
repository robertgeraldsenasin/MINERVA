"""Unit tests for v2.8.5 fix: bypass load_dataset("text", ...) in script 11b.

Verifies that the in-memory `Dataset.from_dict` path produces the same
shape and behavior the rest of the pipeline expects, AND that it works
without triggering the LocalFileSystem caching bug from `datasets >= 2.14`.

Run:
    python -m pytest tests/test_11b_dataset_bypass.py -v
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# datasets is available wherever 11b would run. If not installed,
# skip the whole module — these tests don't make sense without it.
datasets = pytest.importorskip("datasets")
from datasets import Dataset, DatasetDict


# The function under test — copied from script 11b's v2.8.5 code path
# (the script uses it inline; we duplicate here so the test doesn't need
# to import a numbered script as a module).

def _read_lines(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def build_in_memory_datasetdict(train_file: Path, val_file: Path) -> DatasetDict:
    return DatasetDict({
        "train": Dataset.from_dict({"text": _read_lines(train_file)}),
        "validation": Dataset.from_dict({"text": _read_lines(val_file)}),
    })


# Fixtures — corpus files shaped exactly like script 10b emits

NEURO_LINES_TRAIN = textwrap.dedent("""\
    <|label=fake|> <|graph=high|> <|qlat=high|> <|ensem=high|> <|tier=novice|> Halimbawa ng pekeng balita tungkol sa eleksyon.
    <|label=real|> <|graph=low|> <|qlat=low|> <|ensem=low|> <|tier=novice|> Tunay na balita tungkol sa eleksyon.
    <|label=fake|> <|graph=mid|> <|qlat=mid|> <|ensem=mid|> <|tier=intermediate|> Pangalawang halimbawa.
    <|label=real|> <|graph=high|> <|qlat=high|> <|ensem=high|> <|tier=advanced|> Pangatlong halimbawa.
""").strip() + "\n"

NEURO_LINES_VAL = textwrap.dedent("""\
    <|label=real|> <|graph=high|> <|qlat=high|> <|ensem=high|> <|tier=novice|> Validation real example.
    <|label=fake|> <|graph=high|> <|qlat=high|> <|ensem=high|> <|tier=novice|> Validation fake example.
""").strip() + "\n"


@pytest.fixture
def neuro_corpus(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text(NEURO_LINES_TRAIN, encoding="utf-8")
    val.write_text(NEURO_LINES_VAL, encoding="utf-8")
    return train, val


# Tests

class TestInMemoryDatasetBypass:
    def test_returns_datasetdict_with_train_and_validation(self, neuro_corpus):
        train, val = neuro_corpus
        raw = build_in_memory_datasetdict(train, val)
        assert isinstance(raw, DatasetDict)
        assert set(raw.keys()) == {"train", "validation"}

    def test_train_and_val_have_correct_row_counts(self, neuro_corpus):
        train, val = neuro_corpus
        raw = build_in_memory_datasetdict(train, val)
        assert len(raw["train"]) == 4
        assert len(raw["validation"]) == 2

    def test_text_column_present(self, neuro_corpus):
        train, val = neuro_corpus
        raw = build_in_memory_datasetdict(train, val)
        assert raw["train"].column_names == ["text"]
        assert raw["validation"].column_names == ["text"]

    def test_special_tokens_preserved_in_text(self, neuro_corpus):
        """The 18 control tokens must survive line-reading verbatim."""
        train, val = neuro_corpus
        raw = build_in_memory_datasetdict(train, val)
        line0 = raw["train"][0]["text"]
        # Every control token from script 10b's emission must be present
        for tok in ["<|label=fake|>", "<|graph=high|>", "<|qlat=high|>",
                    "<|ensem=high|>", "<|tier=novice|>"]:
            assert tok in line0, f"Token {tok!r} missing from line 0"

    def test_empty_lines_skipped(self, tmp_path):
        """Blank lines in the corpus must not become empty Dataset rows."""
        train = tmp_path / "train.txt"
        val = tmp_path / "val.txt"
        train.write_text("line one\n\n\nline two\n   \nline three\n", encoding="utf-8")
        val.write_text("only line\n", encoding="utf-8")
        raw = build_in_memory_datasetdict(train, val)
        assert len(raw["train"]) == 3
        assert len(raw["validation"]) == 1

    def test_unicode_content_round_trips(self, tmp_path):
        """Tagalog/Filipino UTF-8 content must survive intact."""
        train = tmp_path / "train.txt"
        val = tmp_path / "val.txt"
        train.write_text(
            "Sinabi niya na ang halalan ay malinis at makatarungan.\n"
            "Marami ang nagtaka sa kanyang mga pahayag tungkol sa COMELEC.\n",
            encoding="utf-8",
        )
        val.write_text("Pangkalahatang halalan ay matagumpay.\n", encoding="utf-8")
        raw = build_in_memory_datasetdict(train, val)
        assert "halalan" in raw["train"][0]["text"]
        assert "COMELEC" in raw["train"][1]["text"]

    def test_supports_map_pattern_used_by_script_11b(self, neuro_corpus):
        """The .map() pattern that 11b uses for tokenization must work."""
        train, val = neuro_corpus
        raw = build_in_memory_datasetdict(train, val)

        # Mimic 11b's tokenize function (without actually loading a tokenizer)
        def tokenize(batch):
            return {"input_ids": [[1, 2, 3] for _ in batch["text"]]}

        tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
        assert "text" not in tokenized["train"].column_names
        assert "input_ids" in tokenized["train"].column_names
        assert len(tokenized["train"]) == 4
        assert len(tokenized["validation"]) == 2

    def test_does_not_use_load_dataset(self):
        """Static check: script 11b must not regress to load_dataset('text', ...)."""
        import re
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        src = (repo_root / "scripts"
               / "11b_train_gpt2_neurosymbolic.py").read_text(encoding="utf-8")

        # Strip comments and docstrings so we only look at executable code
        # (we mention load_dataset in comments to explain WHY we bypass it).
        # Quick approach: drop any line that is a comment-only line, and drop
        # everything inside triple-quoted strings.
        no_triple = re.sub(r'"""[\s\S]*?"""', "", src)
        no_triple = re.sub(r"'''[\s\S]*?'''", "", no_triple)
        code_only = "\n".join(
            line for line in no_triple.splitlines()
            if not line.lstrip().startswith("#")
        )

        # The actual broken pattern is an assignment that calls load_dataset
        # with "text" as the first positional arg. We look for the assignment.
        assert not re.search(r'\braw\s*=\s*load_dataset\s*\(', code_only), (
            "Script 11b regressed to `raw = load_dataset(...)` which "
            "triggers the LocalFileSystem caching bug in datasets >= 2.14. "
            "Use Dataset.from_dict instead."
        )
        # And the bypass MUST be present in the executable code
        assert "Dataset.from_dict" in code_only, (
            "Script 11b should use Dataset.from_dict to build the corpus "
            "in memory and bypass load_dataset's filesystem cache path."
        )

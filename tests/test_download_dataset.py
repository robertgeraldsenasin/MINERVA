"""Unit tests for scripts/01_download_dataset.py (v2.8.2).

Covers:
- parse_zip_bytes_to_df handles a JCBlaise-shaped ZIP correctly
- parse_zip_bytes_to_df handles missing inner CSV gracefully
- _save_canonical normalizes "article" -> "text" and "labels" -> "label"
- _save_canonical fails clearly on missing columns
- _save_canonical coerces string label "0"/"1" -> int
- _try_tier_1_direct_zip succeeds when urlopen returns valid ZIP bytes
- _try_tier_1_direct_zip returns None on non-ZIP content (defers to Tier 3)
- _try_tier_1_direct_zip returns None on HTTP errors

These tests run hermetically — no network calls. The actual HF download
is verified by the smoke-test cell in the notebook.

Run:
    python -m pytest tests/test_download_dataset.py -v
"""

from __future__ import annotations

import csv
import importlib.util
import io
import os
import sys
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "01_download_dataset.py"


def _load_module():
    """Load 01_download_dataset.py as a module (module name can't start with digit)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "download_dataset", str(SCRIPT_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["download_dataset"] = mod
    # The script writes data/raw at import time via mkdir; use a tmp cwd
    # so test runs don't pollute the repo.
    spec.loader.exec_module(mod)
    return mod


# Load once at module level — cheap, and side-effects (mkdir) are idempotent.
m = _load_module()


# Fixture: a JCBlaise-shaped ZIP in memory

def _build_fake_jcblaise_zip(rows: list[tuple[str, str]] | None = None,
                             inner_path: str = "fakenews/full.csv") -> bytes:
    """Build an in-memory ZIP shaped like the real JCBlaise download.

    Mirrors the original CSV format: csv.QUOTE_ALL, comma-separated,
    columns ["label", "article"].
    """
    if rows is None:
        rows = [
            ("0", "Tunay na balita tungkol sa eleksyon."),
            ("1", "Pekeng balita na pinapakalat sa social media."),
            ("0", "Resulta ng botohan ayon sa COMELEC."),
            ("1", "Hindi totoong post tungkol kay Candidate A."),
        ]

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf, quoting=csv.QUOTE_ALL)
    writer.writerow(["label", "article"])
    for label, article in rows:
        writer.writerow([label, article])

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner_path, csv_buf.getvalue())
    return zip_buf.getvalue()


# parse_zip_bytes_to_df

class TestParseZipBytes:
    def test_parses_canonical_jcblaise_layout(self):
        zip_bytes = _build_fake_jcblaise_zip()
        df = m.parse_zip_bytes_to_df(zip_bytes)
        assert len(df) == 4
        assert "label" in df.columns
        assert "article" in df.columns
        assert df["label"].tolist() == ["0", "1", "0", "1"]

    def test_handles_quoted_text_with_commas(self):
        rows = [
            ("1", "Sinabi ni Candidate A: hindi totoo, peke lahat ito."),
            ("0", "Ayon sa balita, may bagong patakaran sa COMELEC."),
        ]
        zip_bytes = _build_fake_jcblaise_zip(rows=rows)
        df = m.parse_zip_bytes_to_df(zip_bytes)
        assert len(df) == 2
        assert "Candidate A" in df["article"].iloc[0]
        assert "COMELEC" in df["article"].iloc[1]

    def test_strips_whitespace(self):
        rows = [("0", "  text with leading/trailing spaces  ")]
        zip_bytes = _build_fake_jcblaise_zip(rows=rows)
        df = m.parse_zip_bytes_to_df(zip_bytes)
        assert df["article"].iloc[0] == "text with leading/trailing spaces"

    def test_finds_full_csv_in_alternate_path(self):
        """If the zip's internal layout shifts, parser finds full.csv anywhere."""
        zip_bytes = _build_fake_jcblaise_zip(inner_path="data/full.csv")
        df = m.parse_zip_bytes_to_df(zip_bytes, inner_csv="fakenews/full.csv")
        # Falls back to case-insensitive match
        assert len(df) == 4

    def test_raises_when_no_full_csv_present(self):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("readme.txt", "no csv here")
        with pytest.raises(FileNotFoundError):
            m.parse_zip_bytes_to_df(zip_buf.getvalue())


# _save_canonical

class TestSaveCanonical:
    def test_renames_article_to_text(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Re-create RAW_DIR under tmp
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(m, "RAW_DIR", Path("data/raw"))

        df = pd.DataFrame({
            "article": ["a real article", "a fake one"],
            "label": [0, 1],
        })
        out = m._save_canonical(df, source="test")
        assert out.exists()
        loaded = pd.read_csv(out)
        assert "text" in loaded.columns
        assert "article" not in loaded.columns
        assert loaded["text"].tolist() == ["a real article", "a fake one"]

    def test_renames_labels_to_label(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(m, "RAW_DIR", Path("data/raw"))

        df = pd.DataFrame({
            "text": ["x", "y"],
            "labels": [0, 1],
        })
        out = m._save_canonical(df, source="test")
        loaded = pd.read_csv(out)
        assert "label" in loaded.columns
        assert "labels" not in loaded.columns

    def test_coerces_string_labels_to_int(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(m, "RAW_DIR", Path("data/raw"))

        df = pd.DataFrame({
            "text": ["a", "b", "c"],
            "label": ["0", "1", "0"],  # coming from CSV reader
        })
        out = m._save_canonical(df, source="test")
        loaded = pd.read_csv(out)
        assert loaded["label"].dtype.kind in ("i", "u")
        assert loaded["label"].tolist() == [0, 1, 0]

    def test_raises_on_missing_required_columns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(m, "RAW_DIR", Path("data/raw"))

        df = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError, match="Expected columns"):
            m._save_canonical(df, source="test")


# _try_tier_1_direct_zip — mocked urlopen

class _FakeResponse:
    def __init__(self, content: bytes):
        self._content = content

    def read(self):
        return self._content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestTier1DirectZip:
    def test_succeeds_on_valid_zip(self):
        zip_bytes = _build_fake_jcblaise_zip()
        with patch("urllib.request.urlopen",
                   return_value=_FakeResponse(zip_bytes)):
            df = m._try_tier_1_direct_zip()
        assert df is not None
        assert len(df) == 4
        assert {"label", "article"}.issubset(df.columns)

    def test_returns_none_when_response_is_not_zip(self):
        """If HF starts serving gzip directly, Tier 1 should defer to Tier 3."""
        gzip_bytes = b"\x1f\x8b\x08\x00" + b"\x00" * 100
        with patch("urllib.request.urlopen",
                   return_value=_FakeResponse(gzip_bytes)):
            df = m._try_tier_1_direct_zip()
        assert df is None

    def test_returns_none_on_http_error(self):
        err = urllib.error.HTTPError(
            url=m.JCBLAISE_ZIP_URL, code=404, msg="Not Found",
            hdrs=None, fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=err):
            df = m._try_tier_1_direct_zip()
        assert df is None

    def test_returns_none_on_url_error(self):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("name resolution failed")):
            df = m._try_tier_1_direct_zip()
        assert df is None

    def test_returns_none_on_corrupt_zip(self):
        with patch("urllib.request.urlopen",
                   return_value=_FakeResponse(b"PK\x03\x04not actually a zip")):
            df = m._try_tier_1_direct_zip()
        assert df is None


# End-to-end via Tier 1 (mocked) and canonical save

class TestEndToEndMocked:
    def test_full_flow_produces_valid_csv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(m, "RAW_DIR", Path("data/raw"))

        rows = [(str(i % 2), f"Article number {i} text content") for i in range(20)]
        zip_bytes = _build_fake_jcblaise_zip(rows=rows)

        with patch("urllib.request.urlopen",
                   return_value=_FakeResponse(zip_bytes)):
            df = m._try_tier_1_direct_zip()
            assert df is not None
            out = m._save_canonical(df, source="Tier 1 (direct ZIP)")

        assert out.exists()
        loaded = pd.read_csv(out)

        # Schema downstream scripts (02, 03) require
        assert "text" in loaded.columns
        assert "label" in loaded.columns
        assert loaded["label"].dtype.kind in ("i", "u")
        assert len(loaded) == 20
        assert loaded["label"].sum() == 10  # half of each, balanced

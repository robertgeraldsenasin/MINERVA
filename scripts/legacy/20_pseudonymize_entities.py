from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Allow importing repo-root modules when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minerva_privacy import load_allowed_terms, pseudonymize_texts  # noqa: E402


def _is_json_like(path: Path) -> bool:
    return path.suffix.lower() in {".jsonl", ".json"}


def _load_allowed(path: str | None) -> List[str]:
    if not path:
        return []
    return load_allowed_terms(path)


def pseudonymize_csv(
    in_path: Path,
    out_path: Path,
    text_key: str,
    prefix: str,
    allowed_terms: List[str],
    mapping_out: Path | None,
    report_out: Path | None,
) -> None:
    df = pd.read_csv(in_path)
    if text_key not in df.columns:
        raise KeyError(f"Missing column '{text_key}' in {in_path}")

    texts = [str(t) if isinstance(t, str) else "" for t in df[text_key].tolist()]
    pseudo, mapping = pseudonymize_texts(
        texts,
        placeholder_prefix=prefix,
        allowed_terms=allowed_terms,
    )
    df[text_key] = pseudo

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    if mapping_out is not None:
        mapping_out.parent.mkdir(parents=True, exist_ok=True)
        mapping_out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    if report_out is not None:
        report = {
            "input": str(in_path),
            "output": str(out_path),
            "rows": int(len(df)),
            "text_key": text_key,
            "placeholder_prefix": prefix,
            "allowed_term_count": int(len(allowed_terms)),
            "entities_replaced": int(len(mapping)),
        }
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote pseudonymized CSV -> {out_path}")


def _load_json_records(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """
    Returns (records, was_jsonl)
    """
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows, True

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, False
    if isinstance(payload, dict) and "cards" in payload and isinstance(payload["cards"], list):
        return payload["cards"], False
    raise ValueError(f"Unsupported JSON structure in {path}")


def _write_json_records(path: Path, rows: list[dict[str, Any]], as_jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_jsonl:
        with open(path, "w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def pseudonymize_json(
    in_path: Path,
    out_path: Path,
    text_key: str,
    prefix: str,
    allowed_terms: List[str],
    mapping_out: Path | None,
    report_out: Path | None,
) -> None:
    rows, was_jsonl = _load_json_records(in_path)
    texts = [str(rec.get(text_key, "")) if rec.get(text_key) is not None else "" for rec in rows]

    pseudo, mapping = pseudonymize_texts(
        texts,
        placeholder_prefix=prefix,
        allowed_terms=allowed_terms,
    )

    for rec, t in zip(rows, pseudo):
        rec[text_key] = t
        rec["pseudonymized"] = True
        rec["placeholder_prefix"] = prefix
        rec["candidate_allowlist_applied"] = bool(allowed_terms)

    _write_json_records(out_path, rows, as_jsonl=was_jsonl)

    if mapping_out is not None:
        mapping_out.parent.mkdir(parents=True, exist_ok=True)
        mapping_out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    if report_out is not None:
        report = {
            "input": str(in_path),
            "output": str(out_path),
            "rows": int(len(rows)),
            "text_key": text_key,
            "placeholder_prefix": prefix,
            "allowed_term_count": int(len(allowed_terms)),
            "entities_replaced": int(len(mapping)),
            "json_mode": "jsonl" if was_jsonl else "json",
        }
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote pseudonymized {'JSONL' if was_jsonl else 'JSON'} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pseudonymize person-like entities in CSV/JSONL/JSON while preserving candidate allowlist names."
    )
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--placeholder_prefix", default="Entity")
    ap.add_argument("--allowlist_json", default=None, help="Candidate profiles JSON whose names/aliases should be preserved.")
    ap.add_argument("--mapping_out", default=None, help="Optional JSON path for entity->placeholder mapping.")
    ap.add_argument("--report_out", default=None, help="Optional JSON path for pseudonymization report.")
    args = ap.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    allowed_terms = _load_allowed(args.allowlist_json)
    mapping_out = Path(args.mapping_out) if args.mapping_out else None
    report_out = Path(args.report_out) if args.report_out else None

    if _is_json_like(in_path):
        pseudonymize_json(
            in_path,
            out_path,
            args.text_key,
            args.placeholder_prefix,
            allowed_terms,
            mapping_out,
            report_out,
        )
    else:
        pseudonymize_csv(
            in_path,
            out_path,
            args.text_key,
            args.placeholder_prefix,
            allowed_terms,
            mapping_out,
            report_out,
        )


if __name__ == "__main__":
    main()

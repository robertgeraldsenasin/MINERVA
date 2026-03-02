from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow importing repo-root modules when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minerva_privacy import pseudonymize_texts  # noqa: E402


def _is_jsonl(path: Path) -> bool:
    return path.suffix.lower() in {".jsonl", ".json"}


def pseudonymize_csv(in_path: Path, out_path: Path, text_key: str, prefix: str) -> None:
    df = pd.read_csv(in_path)
    if text_key not in df.columns:
        raise KeyError(f"Missing column '{text_key}' in {in_path}")
    texts = [str(t) if isinstance(t, str)
             else "" for t in df[text_key].tolist()]
    pseudo, _ = pseudonymize_texts(texts, placeholder_prefix=prefix)
    df[text_key] = pseudo
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote pseudonymized CSV -> {out_path}")


def pseudonymize_jsonl(in_path: Path, out_path: Path, text_key: str, prefix: str) -> None:
    rows = []
    texts = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(rec)
            texts.append(str(rec.get(text_key, ""))
                         if rec.get(text_key) is not None else "")

    pseudo, _ = pseudonymize_texts(texts, placeholder_prefix=prefix)
    for rec, t in zip(rows, pseudo):
        rec[text_key] = t
        rec["pseudonymized"] = True
        rec["placeholder_prefix"] = prefix

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote pseudonymized JSONL -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pseudonymize person-like entities in CSV/JSONL.")
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--placeholder_prefix", default="Candidate")
    args = ap.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    if _is_jsonl(in_path):
        pseudonymize_jsonl(in_path, out_path, args.text_key,
                           args.placeholder_prefix)
    else:
        pseudonymize_csv(in_path, out_path, args.text_key,
                         args.placeholder_prefix)


if __name__ == "__main__":
    main()

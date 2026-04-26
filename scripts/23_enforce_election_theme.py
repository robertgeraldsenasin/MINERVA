from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Allow importing repo-root modules when running `python scripts/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minerva_privacy import pseudonymize_texts  # noqa: E402

DEFAULT_ELECTION_KEYWORDS = [
    "election", "eleksyon", "campaign", "kampanya", "candidate", "kandidato", "vote",
    "boto", "balota", "ballot", "survey", "poll", "debate", "rally", "endorsement",
    "platform", "plataporma", "mayor", "governor", "senator", "representative",
    "councilor", "councillor", "vice mayor", "vice-mayor", "precinct", "canvassing",
    "turnout", "political", "party", "ticket", "speech", "town hall", "donation",
    "poll watcher", "proclamation",
]


def read_json_or_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
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


def write_json_or_jsonl(path: Path, rows: list[dict[str, Any]], as_jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_jsonl:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def load_candidate_profiles(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        # accept {"A": {...}, "B": {...}} or {"candidates": [...]}
        if "candidates" in payload and isinstance(payload["candidates"], list):
            out = {}
            for item in payload["candidates"]:
                cid = str(item.get("candidate_id") or item.get("code") or item.get("id"))
                if cid:
                    out[cid] = dict(item)
            return out
        return {str(k): dict(v) for k, v in payload.items()}

    if isinstance(payload, list):
        out = {}
        for item in payload:
            cid = str(item.get("candidate_id") or item.get("code") or item.get("id"))
            if cid:
                out[cid] = dict(item)
        return out

    raise ValueError("Unsupported candidate profile JSON structure.")


def extract_aliases(profile_key: str, profile: dict[str, Any]) -> list[str]:
    vals: list[str] = [profile_key]
    for k in ("name", "display_name", "public_name", "short_name"):
        val = profile.get(k)
        if isinstance(val, str) and val.strip():
            vals.append(val.strip())
    aliases = profile.get("aliases", [])
    if isinstance(aliases, list):
        vals.extend([str(a).strip() for a in aliases if str(a).strip()])
    # Remove duplicates while preserving order
    seen = set()
    out = []
    for v in vals:
        low = v.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(v)
    return out


def detect_targets(text: str, profiles: dict[str, dict[str, Any]]) -> list[str]:
    text = text or ""
    hits: list[str] = []
    low = text.lower()
    for cid, prof in profiles.items():
        aliases = extract_aliases(cid, prof)
        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue
            if re.search(rf"(?<!\w){re.escape(alias.lower())}(?!\w)", low):
                hits.append(cid)
                break
    return hits


def find_election_terms(text: str, keywords: Sequence[str]) -> list[str]:
    low = (text or "").lower()
    hits = []
    for kw in keywords:
        if re.search(rf"(?<!\w){re.escape(kw.lower())}(?!\w)", low):
            hits.append(kw)
    return hits


def all_candidate_terms(profiles: dict[str, dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for cid, prof in profiles.items():
        out.extend(extract_aliases(cid, prof))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter generated/scored cards to election-themed, three-candidate-focused content and pseudonymize irrelevant names."
    )
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--candidate_profiles", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--election_keywords", default=None, help="Optional JSON file containing a list of election keywords.")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--placeholder_prefix", default="Entity")
    ap.add_argument("--require_candidate", action="store_true", default=True)
    ap.add_argument("--no_require_candidate", dest="require_candidate", action="store_false")
    ap.add_argument("--require_election_term", action="store_true", default=True)
    ap.add_argument("--no_require_election_term", dest="require_election_term", action="store_false")
    ap.add_argument("--single_target_only", action="store_true", help="Keep only cards focusing on exactly one candidate.")
    ap.add_argument("--report_out", default=None)
    args = ap.parse_args()

    in_path = Path(args.in_file)
    profile_path = Path(args.candidate_profiles)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if not profile_path.exists():
        raise FileNotFoundError(profile_path)

    rows, as_jsonl = read_json_or_jsonl(in_path)
    profiles = load_candidate_profiles(profile_path)

    if args.election_keywords:
        keyword_path = Path(args.election_keywords)
        keywords = json.loads(keyword_path.read_text(encoding="utf-8"))
        if isinstance(keywords, dict) and "keywords" in keywords:
            keywords = keywords["keywords"]
        if not isinstance(keywords, list):
            raise ValueError("Election keyword JSON must be a list or {'keywords': [...]} object.")
        keywords = [str(x).strip() for x in keywords if str(x).strip()]
    else:
        keywords = list(DEFAULT_ELECTION_KEYWORDS)

    allowed_terms = all_candidate_terms(profiles)
    out_rows: list[dict[str, Any]] = []
    rejected = 0

    for row in rows:
        text = str(row.get(args.text_key, "") or "")
        targets = detect_targets(text, profiles)
        election_hits = find_election_terms(text, keywords)

        keep = True
        if args.require_candidate and not targets:
            keep = False
        if args.require_election_term and not election_hits:
            keep = False
        if args.single_target_only and len(targets) != 1:
            keep = False

        if not keep:
            rejected += 1
            continue

        pseudo, mapping = pseudonymize_texts(
            [text],
            placeholder_prefix=args.placeholder_prefix,
            allowed_terms=allowed_terms,
        )

        row = dict(row)
        row[args.text_key] = pseudo[0]
        row["targets"] = targets
        row["theme_flags"] = {
            "is_on_theme": True,
            "candidate_targets": targets,
            "candidate_focus": "single" if len(targets) == 1 else ("multi" if len(targets) > 1 else "none"),
            "election_keywords": election_hits,
        }
        if mapping:
            row["redacted_entities"] = mapping
        out_rows.append(row)

    out_path = Path(args.out_file)
    write_json_or_jsonl(out_path, out_rows, as_jsonl=as_jsonl)

    if args.report_out:
        report = {
            "input": str(in_path),
            "output": str(out_path),
            "input_rows": int(len(rows)),
            "kept_rows": int(len(out_rows)),
            "rejected_rows": int(rejected),
            "require_candidate": bool(args.require_candidate),
            "require_election_term": bool(args.require_election_term),
            "single_target_only": bool(args.single_target_only),
            "candidate_count": int(len(profiles)),
            "keyword_count": int(len(keywords)),
        }
        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote themed card set -> {out_path} (kept={len(out_rows)} rejected={rejected})")


if __name__ == "__main__":
    main()

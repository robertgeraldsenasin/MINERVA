#!/usr/bin/env python3
"""Extract real-name surface forms from the JCBlaise training corpus into a blocklist."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_TITLE = (
    r"(?:Mr|Mrs|Ms|Miss|Dr|Sen|Senator|Mayor|President|VP|"
    r"Vice\s+President|Gov|Governor|Rep|Representative|Cong|"
    r"Congressman|Congresswoman|Councilor|Councillor|Atty|Hon|Honorable)"
)
_PARTICLE = r"(?:si|kay|kina|ni|ng|ang)"
_CAPWORD = r"[A-Z][a-z][A-Za-zÀ-ÿ'’-]{1,}"
_MULTIWORD = rf"{_CAPWORD}(?:\s+{_CAPWORD}){{1,3}}"

_DETECT_PATTERNS = [
    re.compile(rf"\b{_TITLE}\.?\s+({_MULTIWORD})\b"),
    re.compile(rf"\b{_PARTICLE}\s+({_MULTIWORD})\b"),
    re.compile(rf"\b({_MULTIWORD})\b"),
]

_NON_NAMES = {
    "Ang", "Ng", "Mga", "Ito", "Iyon", "Iyan", "Hindi", "Oo",
    "The", "This", "That", "These", "Those", "It", "He", "She", "They",
    "Breaking", "Update", "Report", "News", "Statement",
    "Balita", "Ulat", "Trending", "Viral", "Babala", "Paalala",
    "Election", "Campaign", "Survey", "Debate", "Result", "Results",
    "Official", "Department", "Office", "Council", "Senate", "Congress",
    "Committee", "Court", "Commission", "Bureau", "Agency",
    "Philippines", "Manila", "Cebu", "Davao", "Quezon", "Makati",
    "Pampanga", "Bulacan", "Rizal", "Tagum", "Bicol",
    "Mabilis", "Bukas", "Lunes", "Martes", "Miyerkules",
    "Huwebes", "Biyernes", "Sabado", "Linggo",
    "Enero", "Pebrero", "Marso", "Abril", "Mayo", "Hunyo", "Hulyo",
    "Agosto", "Setyembre", "Oktubre", "Nobyembre", "Disyembre",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Facebook", "Twitter", "YouTube", "TikTok", "Instagram",
    "WhatsApp", "Viber", "Telegram", "Messenger",
    "Filipino", "Pilipino", "Tagalog", "English", "American",
    "Christian", "Muslim", "Catholic",
}


def extract_from_text(text: str) -> list[str]:
    names: list[str] = []
    if not isinstance(text, str):
        return names
    for pat in _DETECT_PATTERNS:
        for m in pat.finditer(text):
            ent = m.group(1).strip()
            if not ent:
                continue
            tokens = ent.split()
            if all(t in _NON_NAMES for t in tokens):
                continue
            if len(tokens) == 1 and tokens[0] in _NON_NAMES:
                continue
            names.append(ent)
    return names


def load_jcblaise_online() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "datasets library not installed. Run: pip install datasets\n"
            "Or use --cached <path-to-csv>."
        )
    logger.info("Downloading jcblaise/fake_news_filipino from HuggingFace...")
    ds = load_dataset("jcblaise/fake_news_filipino")
    rows = []
    for split_name, split_data in ds.items():
        for item in split_data:
            text = item.get("article") or item.get("text") or ""
            label = item.get("label", -1)
            rows.append({"split": split_name, "text": text, "label": label})
    logger.info("Total rows: %d", len(rows))
    return rows


def load_jcblaise_cached(cached_path: str) -> list[dict]:
    import csv
    rows = []
    with open(cached_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = r.get("article") or r.get("text") or ""
            label = r.get("label", "-1")
            rows.append({"split": "cached", "text": text, "label": label})
    logger.info("Loaded %d rows from %s", len(rows), cached_path)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Extract names from JCBlaise dataset.")
    p.add_argument("--out_file", default="templates/jcblaise_real_names_blocklist.txt")
    p.add_argument("--report_out", default="reports/jcblaise_extraction.json")
    p.add_argument("--min_count", type=int, default=3)
    p.add_argument("--cached", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    rows = load_jcblaise_cached(args.cached) if args.cached else load_jcblaise_online()
    if args.limit > 0:
        rows = rows[:args.limit]
        logger.info("Limited to first %d rows", len(rows))

    name_counts: Counter = Counter()
    name_per_label: dict[str, Counter] = {"0": Counter(), "1": Counter()}
    for row in rows:
        text = row.get("text") or ""
        label = str(row.get("label", "-1"))
        for n in extract_from_text(text):
            n_norm = n.lower().strip()
            name_counts[n_norm] += 1
            if label in name_per_label:
                name_per_label[label][n_norm] += 1

    blocklisted = [n for n, c in name_counts.most_common() if c >= args.min_count]
    logger.info("%d unique names; %d meet min_count=%d threshold",
                len(name_counts), len(blocklisted), args.min_count)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# JCBlaise Real-Names Blocklist",
        f"# Generated: {datetime.now(timezone.utc).isoformat()}",
        "# Source: jcblaise/fake_news_filipino (Cruz et al., 2020)",
        f"# Min occurrence threshold: {args.min_count}",
        f"# Total names: {len(blocklisted)}",
        "#",
        "# Used by scripts/33_strict_name_allowlist.py to block any of",
        "# these real-PH names from leaking into game card output.",
        "# Format: one lowercase name per line. Blank lines and lines",
        "# starting with # are ignored.",
        "",
    ]
    out_path.write_text("\n".join(header) + "\n".join(blocklisted) + "\n",
                        encoding="utf-8")
    logger.info("Wrote blocklist -> %s", out_path)

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": "jcblaise/fake_news_filipino",
        "citation": "Cruz, Tan, & Cheng (2020). Localization of Fake News Detection via Multitask Transfer Learning. LREC 2020.",
        "rows_processed": len(rows),
        "min_count": args.min_count,
        "unique_names_total": len(name_counts),
        "names_blocklisted": len(blocklisted),
        "top_50_names_overall": [{"name": n, "count": c}
                                 for n, c in name_counts.most_common(50)],
        "top_30_in_fake_articles": [{"name": n, "count": c}
                                    for n, c in name_per_label.get("1", Counter()).most_common(30)],
        "top_30_in_real_articles": [{"name": n, "count": c}
                                    for n, c in name_per_label.get("0", Counter()).most_common(30)],
        "output_blocklist": str(out_path),
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    logger.info("=" * 60)
    logger.info("JCBlaise extraction complete")
    logger.info("  Rows processed     : %d", len(rows))
    logger.info("  Unique names found : %d", len(name_counts))
    logger.info("  Names blocklisted  : %d", len(blocklisted))
    logger.info("  Top 5 in dataset:")
    for n, c in name_counts.most_common(5):
        logger.info("    - %s (%d)", n, c)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
33_strict_name_allowlist.py  -  v2.6.final

POST-PIPELINE STRICT NAME-ALLOWLIST ENFORCER

Per the user's v2.6.final decisions:
  - Generic candidate naming only ("Candidate A/B/C")
  - Add an explicit allowlist enforcer that blocks ANY non-allowed name

This is the LAST LINE OF DEFENSE before cards reach the Unity game.
After templates (script 25/30), after pseudonymization (scripts 22/31),
after theme enforcement (script 23), after curation (script 24) - this
script audits every card text and either:

  1. REJECTS the card (if it contains person-like names not on the
     candidate allowlist), or
  2. REDACTS the foreign names with a placeholder, or
  3. PASSES the card through (no foreign names found).

CITATIONS
---------
- Pilan, I., et al. (2022). The Text Anonymization Benchmark (TAB).
  Computational Linguistics, 48(4), 1053-1101.
- Yermilov, O., Raheja, V., & Chernodub, A. (2023). Privacy- and
  Utility-Preserving NLP with Anonymized Data. EACL 2023.
- Roozenbeek, J., & van der Linden, S. (2019). The fake news game.
  Palgrave Communications, 5(1).
- Cruz, J. C. B., Tan, J. A., & Cheng, C. K. (2020). Localization of
  Fake News Detection via Multitask Transfer Learning. LREC 2020.
- BATB_CompiledThesisPaper section 1.5 Limitation #2.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_TITLE = (
    r"(?:Mr|Mrs|Ms|Miss|Dr|Sen|Senator|Mayor|President|VP|"
    r"Vice\s+President|Gov|Governor|Rep|Representative|Cong|"
    r"Congressman|Congresswoman|Councilor|Councillor|Atty|Hon|Honorable)"
)
_PARTICLE_PERSON = r"(?:si|kay|kina|ni)"
_CAPWORD = r"[A-Z][a-z][A-Za-zÀ-ÿ'’-]{1,}"
_NAME_TOKEN = rf"{_CAPWORD}(?:\s+{_CAPWORD}){{0,3}}"
_MULTIWORD = rf"{_CAPWORD}(?:\s+{_CAPWORD}){{1,3}}"

_DETECT_PATTERNS = [
    ("title_name",          re.compile(rf"\b{_TITLE}\.?\s+({_NAME_TOKEN})\b"), 1),
    ("particle_title_name", re.compile(rf"\b{_PARTICLE_PERSON}\s+(?:{_TITLE}\.?\s+)?({_NAME_TOKEN})\b"), 1),
    ("multiword_alone",     re.compile(rf"\b({_MULTIWORD})\b"), 1),
    ("said_surname",        re.compile(rf"\b({_CAPWORD})\b\s+(?:said|sabi|ayon|nagsalita|ipinahayag|ipinakita|pahayag)"), 1),
    ("according_to",        re.compile(rf"\b(?:According to|according to|Ayon kay|ayon kay|Sinabi ni|sinabi ni)\s+(?:{_TITLE}\.?\s+)?({_NAME_TOKEN})\b"), 1),
    ("comma_attribution",   re.compile(rf",\s+({_CAPWORD})\s*,"), 1),
]

_DEFINITE_NON_NAMES = {
    # Tagalog function / sentence-start words
    "Ang", "Ng", "Mga", "Ito", "Iyon", "Iyan", "Hindi", "Oo", "Sige",
    "Pero", "Subalit", "Ngunit", "Datapwat", "Kahit", "Kapag", "Kung",
    "Walang", "May", "Mayroon", "Wala", "Marami", "Maraming",
    "Sa", "Para", "Mula", "Hanggang", "Bago", "Pagkatapos",
    "Ako", "Ikaw", "Siya", "Kami", "Tayo", "Sila", "Niya", "Nila",
    # English equivalents
    "The", "This", "That", "These", "Those", "It", "He", "She", "They",
    "Today", "Tomorrow", "Yesterday", "Tonight", "Now", "Then",
    "Breaking", "Update", "Report", "News", "Statement",
    # Tagalog news intros
    "Balita", "Ulat", "Trending", "Viral", "Babala", "Paalala",
    "Tingnan", "Pansinin", "Importante", "Kahapon", "Ngayon",
    # Generic narrative
    "Election", "Campaign", "Survey", "Debate", "Result", "Results",
    "Official", "Department", "Office", "Council", "Senate", "Congress",
    "Committee", "Court", "Commission", "Bureau", "Agency",
    # Philippine cities and provinces
    "Philippines", "Pilipinas", "Manila", "Maynila", "Cebu", "Davao",
    "Quezon", "Makati", "Pampanga", "Bulacan", "Laguna", "Batangas",
    "Cavite", "Pangasinan", "Iloilo", "Bacolod", "Cagayan", "Zamboanga",
    "Rizal", "Tagum", "Baguio", "Tarlac", "Bicol", "Visayas", "Mindanao",
    "Luzon", "Antipolo", "Pasig", "Mandaluyong", "Marikina", "Caloocan",
    "Taguig", "Paranaque", "Las Pinas", "Muntinlupa", "Pasay", "Valenzuela",
    "Malabon", "Navotas",
    # Days
    "Lunes", "Martes", "Miyerkules", "Huwebes", "Biyernes", "Sabado",
    "Linggo", "Mabilis",
    # Months (TL + EN)
    "Enero", "Pebrero", "Marso", "Abril", "Mayo", "Hunyo", "Hulyo",
    "Agosto", "Setyembre", "Oktubre", "Nobyembre", "Disyembre",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    # Platforms
    "Facebook", "Twitter", "YouTube", "TikTok", "Instagram", "WhatsApp",
    "Viber", "Telegram", "Messenger", "Line", "WeChat",
    # Adjectives
    "Filipino", "Pilipino", "Tagalog", "English", "American",
    "Christian", "Muslim", "Catholic",
    # Titles that should never be treated as names by themselves
    "Atty", "Hon", "Dr", "Mr", "Mrs", "Ms", "Sen", "Rep", "Cong", "Gov",
    "VP", "Sec", "Engr", "Prof",
    # Tagalog institutional words
    "Senado", "Kongreso", "Korte", "Komisyon", "Kagawaran", "Tanggapan",
    "Lungsod", "Bayan", "Probinsya", "Lalawigan", "Munisipalidad",
    "Barangay", "Kapitolyo", "Malacañang", "Malacanang",
    # The literal token "Candidate" (it pairs with A/B/C in the allowlist
    # but appearing alone is fine)
    "Candidate",
}

_ALLOWED_ORGANIZATIONS = {
    # Real PH fact-checkers and credible news organizations
    "vera files", "rappler", "pcij",
    "philippine star", "philstar", "abs-cbn", "gma news", "inquirer",
    "philippine news agency", "pna", "philippine daily inquirer",
    "comelec", "namfrel", "lente",
    "international fact-checking network", "ifcn",
    # PH polling / research firms
    "pulse asia", "social weather stations", "sws", "stratbase",
    # Multi-word PH city names (all generic locations)
    "quezon city", "cebu city", "davao city", "manila city",
    "makati city", "pasig city", "taguig city", "paranaque city",
    "las pinas city", "muntinlupa city", "caloocan city",
    "san juan city", "valenzuela city", "marikina city",
    "mandaluyong city", "malabon city", "navotas city",
    "bacolod city", "iloilo city", "zamboanga city",
    "cagayan de oro", "general santos", "metro manila",
    # Fictional / template-introduced organisations
    "pilipinas truth watch", "truth-bureau", "city election bulletin",
    # Generic party names in v2.6.final
    "party a", "party b", "party c",
    # v2.9.7: public-domain PH government / institutional / news / geographic
    # terms that surface legitimately in GPT-2 cards. These are generic
    # references (not naming any individual person), so they don't violate
    # the pseudonymization principle. Adding them prevents the strict
    # allowlist from rejecting otherwise-clean cards on these terms.
    # Government institutions:
    "supreme court", "court of appeals", "sandiganbayan", "ombudsman",
    "department of justice", "doj", "department of education", "deped",
    "the deped", "department of health", "doh", "department of foreign affairs", "dfa",
    "department of interior and local government", "dilg",
    "department of national defense", "dnd", "department of finance", "dof",
    "department of trade and industry", "dti",
    "department of public works and highways", "dpwh",
    "department of transportation", "dotr", "department of agriculture", "da",
    "armed forces of the philippines", "afp", "philippine national police", "pnp",
    "presidential communications operations office", "pcoo",
    "presidential communications office", "pco",
    "national bureau of investigation", "nbi",
    "national disaster risk reduction and management council", "ndrrmc",
    "national economic and development authority", "neda",
    "bureau of internal revenue", "bir",
    "philippine institute", "philippine institute for development studies",
    "police regional office", "regional trial court", "rtc",
    # Geographic / administrative:
    "capital metro area", "capital metro area council",
    "island group", "china sea", "west philippine sea",
    "barangay sta", "barangay sto", "barangay san",
    "antipolo city", "tuguegarao city", "olongapo city",
    "tagaytay city", "tarlac city", "san fernando city",
    "city hall", "city hospital",
    # Generic news/media generics:
    "daily news", "evening news", "morning news",
    # Generic titles (lowercased; they appear as words in posts):
    "justice", "papa",
    # Geographic / country generics (when discussed as topics, not people):
    "china",
    # v2.9.7 additions: composite forms that the parser groups as single tokens.
    # These appear when a government-role title precedes "Candidate A/B/C" and
    # the entity extractor concatenates them. Adding them prevents false positives.
    "deped candidate", "dilg candidate", "doj candidate", "doh candidate",
    "doj official", "deped official", "comelec official",
    "philippine institute of volcanology and seismology", "phivolcs",
    "philippine atmospheric geophysical and astronomical services administration",
    "pagasa",
    # v2.9.8 additions: 13 edge cases surfaced by the v2.9.7 run zip.
    # These are generic role-titles or law-enforcement units that GPT-2
    # uses descriptively (not as proper names of any individual).
    "the president", "president", "the senator", "senator", "the mayor",
    "the governor", "the secretary", "secretary",
    "police district", "police station", "police force",
    "chief gen", "chief general", "general", "the general",
    "politiko hindi", "politiko", "the politician", "politician",
    "the cabinet", "cabinet",
    "the senate", "the congress", "congressman", "congresswoman",
    # v2.9.9 additions: 8 final edge cases from the v2.9.8 run zip.
    # Categorized:
    #   - Generic role titles in Tagalog/English:
    "presidente", "presidential", "education sec", "education secretary",
    "the education sec",
    #   - PNP / agency unit names (generic, not proper):
    "intelligence group", "pnp intelligence group", "intelligence service",
    "investigation group", "investigation division",
    #   - Real GMA/ABS-CBN/Inquirer news program / outlet titles (these
    #     are press citations, not "candidate names" — same status as
    #     existing entries like "rappler", "inquirer"):
    "unang balita", "24 oras", "tv patrol", "saksi",
    #   - Filipino generic gendered/colloquial terms (NOT real names):
    "isang pinay", "pinay", "pinoy", "isang pinoy",
    #   - Foreign nationalities used adjectively in news context (the
    #     persons referenced are still pseudonymized as Candidate A/B/C;
    #     the nationality adjective itself is descriptive, not naming):
    "japanese", "the japan", "japan", "indonesia", "indonesian",
    "korean", "the korea", "korea", "chinese vlogger", "japanese vlogger",
}

# Used as fallback when the JCBlaise blocklist file is missing.
# Sourced from documented PH political dynasties (Arugay & Baquisal 2022,
# Mendoza et al. 2012, BATB section 1.5).
_FALLBACK_BLOCKLIST = {
    "marcos", "duterte", "aquino", "estrada", "arroyo", "macapagal",
    "binay", "robredo", "roxas", "pacquiao", "lacson", "poe",
    "bbm", "leni", "isko", "moreno", "pangilinan", "sotto", "drilon",
    "enrile", "lapid", "pimentel", "cayetano", "trillanes", "honasan",
    "villar", "revilla", "angara", "escudero", "gordon", "recto",
    "legarda", "ejercito",
    "ferdinand marcos", "rodrigo duterte", "ninoy aquino", "noynoy aquino",
    "cory aquino", "gloria arroyo", "manny pacquiao",
    "leni robredo", "isko moreno", "panfilo lacson", "grace poe",
    "tito sotto", "miriam santiago", "miriam defensor",
    "bongbong marcos",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_allowlist(profiles_path: str | Path) -> set[str]:
    p = Path(profiles_path)
    if not p.exists():
        raise FileNotFoundError(f"Profile path not found: {p}")
    if p.suffix == ".py":
        import importlib.util
        spec = importlib.util.spec_from_file_location("_cand_cfg", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = getattr(mod, "CANDIDATES_CONFIG", None)
        if cfg is None:
            raise ValueError(f"Python module {p} has no CANDIDATES_CONFIG")
        items = cfg
    else:
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "candidates" in payload:
            items = payload["candidates"]
        elif isinstance(payload, dict):
            items = list(payload.values())
        elif isinstance(payload, list):
            items = payload
        else:
            raise ValueError(f"Unsupported profile JSON: {type(payload)}")
    allowed: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("name", "display_name", "public_name", "short_name",
                    "first_name", "last_name", "nickname"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                allowed.add(_norm(val))
        first = item.get("first_name", "")
        last = item.get("last_name", "")
        if first and last:
            allowed.add(_norm(f"{first} {last}"))
        aliases = item.get("aliases") or []
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    allowed.add(_norm(a))
    for code in ("a", "b", "c"):
        allowed.add(f"candidate {code}")
    return allowed


def load_blocklist(blocklist_path: str | Path | None) -> set[str]:
    if blocklist_path and Path(blocklist_path).exists():
        lines = Path(blocklist_path).read_text(encoding="utf-8").splitlines()
        names = {_norm(ln) for ln in lines if ln.strip() and not ln.startswith("#")}
        names |= _FALLBACK_BLOCKLIST
        return names
    logger.warning(
        "Blocklist file not found at %s; using built-in fallback only. "
        "Run scripts/34_extract_jcblaise_names.py on Colab.", blocklist_path,
    )
    return set(_FALLBACK_BLOCKLIST)


def detect_person_spans(text: str) -> list[tuple[int, int, str, str]]:
    spans: list[tuple[int, int, str, str]] = []
    for pat_name, pat, grp in _DETECT_PATTERNS:
        for m in pat.finditer(text):
            try:
                s, e = m.span(grp)
                ent = m.group(grp)
            except (IndexError, re.error):
                continue
            if not ent:
                continue
            # Skip if the whole entity is a definite non-name
            if ent in _DEFINITE_NON_NAMES:
                continue
            # Skip if ALL tokens of a multi-word match are non-names
            # (e.g., "Today Tomorrow" or "Pero Walang")
            tokens = ent.split()
            if all(t in _DEFINITE_NON_NAMES for t in tokens):
                continue
            spans.append((s, e, ent, pat_name))
    seen = set()
    out = []
    for s, e, ent, pat_name in sorted(spans, key=lambda x: (x[0], -x[1])):
        key = (s, e, ent.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((s, e, ent, pat_name))
    return out


def detect_blocklist_tokens(text: str, blocked: set[str]) -> list[tuple[int, int, str, str]]:
    out: list[tuple[int, int, str, str]] = []
    if not blocked:
        return out
    single_word = sorted(
        {b for b in blocked if " " not in b and len(b) >= 4},
        key=lambda x: -len(x),
    )
    if not single_word:
        return out
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in single_word) + r")\b",
        flags=re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        s, e = m.span(1)
        ent = m.group(1)
        out.append((s, e, ent, "blocklist_direct"))
    return out


def classify_spans(spans, allowed: set[str], blocked: set[str]):
    allowed_out = []
    foreign_out = []
    for s, e, ent, pat in spans:
        norm = _norm(ent)
        toks = norm.split()
        is_allowed = norm in allowed or any(
            " ".join(toks[i:j]) in allowed
            for i in range(len(toks))
            for j in range(i + 1, len(toks) + 1)
        )
        if is_allowed:
            allowed_out.append((s, e, ent, pat))
            continue
        is_org = norm in _ALLOWED_ORGANIZATIONS or any(
            " ".join(toks[i:j]) in _ALLOWED_ORGANIZATIONS
            for i in range(len(toks))
            for j in range(i + 1, len(toks) + 1)
        )
        if is_org:
            allowed_out.append((s, e, ent, pat))
            continue
        is_blocked = norm in blocked or any(t in blocked for t in toks)
        if is_blocked:
            foreign_out.append((s, e, ent, pat, "blocklist"))
            continue
        if len(toks) == 1 and len(toks[0]) >= 4:
            foreign_out.append((s, e, ent, pat, "unknown_name"))
        elif len(toks) >= 2:
            foreign_out.append((s, e, ent, pat, "unknown_name"))
    return allowed_out, foreign_out


def redact_text(text: str, foreign_spans, redaction: str) -> str:
    sorted_spans = sorted(foreign_spans, key=lambda x: (x[0], -(x[1] - x[0])))
    selected = []
    last_end = -1
    for s, e, ent, pat, reason in sorted_spans:
        if s < last_end:
            continue
        selected.append((s, e, ent, pat, reason))
        last_end = e
    out = text
    for s, e, ent, pat, reason in sorted(selected, key=lambda x: -x[0]):
        out = out[:s] + redaction + out[e:]
    return out


def process_card(card, allowed, blocked, mode, redaction, text_key):
    text = card.get(text_key, "") or ""
    spans = detect_person_spans(text)
    allowed_spans, foreign_spans = classify_spans(spans, allowed, blocked)
    direct_hits = detect_blocklist_tokens(text, blocked)
    for s, e, ent, pat in direct_hits:
        if _norm(ent) in allowed:
            continue
        if _norm(ent) in _ALLOWED_ORGANIZATIONS:
            continue
        if any(s >= fs and e <= fe for fs, fe, _, _, _ in foreign_spans):
            continue
        if any(s >= fs and e <= fe for fs, fe, _, _ in allowed_spans):
            continue
        foreign_spans.append((s, e, ent, pat, "blocklist"))
    if not foreign_spans:
        return True, card, []
    if mode == "reject":
        return False, card, foreign_spans
    new_text = redact_text(text, foreign_spans, redaction)
    new_card = dict(card)
    new_card[text_key] = new_text
    new_card["strict_allowlist_redactions"] = len(foreign_spans)
    return True, new_card, foreign_spans


def main() -> None:
    p = argparse.ArgumentParser(description="v2.6.final strict allowlist enforcer.")
    p.add_argument("--in_file", required=True)
    p.add_argument("--candidate_profiles", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--report_out", required=True)
    p.add_argument("--blocklist_file", default="templates/jcblaise_real_names_blocklist.txt")
    p.add_argument("--mode", default="reject", choices=["reject", "redact"])
    p.add_argument("--redaction", default="[Iba pang tao]")
    p.add_argument("--text_key", default="text")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    allowed = load_allowlist(args.candidate_profiles)
    blocked = load_blocklist(args.blocklist_file)
    logger.info("Allowlist size: %d entries", len(allowed))
    logger.info("Blocklist size: %d entries", len(blocked))
    payload = json.loads(Path(args.in_file).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "cards" in payload:
        cards = payload["cards"]; wrapped = True; meta = payload.get("_metadata", {})
    else:
        cards = payload; wrapped = False; meta = {}
    logger.info("Loaded %d cards from %s", len(cards), args.in_file)
    kept = []; rejected = []; redaction_count = 0
    foreign_counter: Counter = Counter()
    foreign_examples: list[dict] = []
    for card in cards:
        keep, mod_card, foreign = process_card(
            card, allowed, blocked, args.mode, args.redaction, args.text_key
        )
        for s, e, ent, pat, reason in foreign:
            foreign_counter[(_norm(ent), reason)] += 1
            if len(foreign_examples) < 50:
                foreign_examples.append({
                    "card_id": card.get("id", "?"),
                    "name": ent, "pattern": pat, "reason": reason,
                    "context": card.get(args.text_key, "")[max(0, s-30):e+30],
                })
        if keep:
            kept.append(mod_card)
            if args.mode == "redact" and foreign:
                redaction_count += len(foreign)
        else:
            rejected.append({"id": card.get("id", "?")})
    if wrapped:
        out_payload = {"_metadata": dict(meta), "cards": kept}
        out_payload["_metadata"]["strict_allowlist_applied"] = True
        out_payload["_metadata"]["strict_allowlist_mode"] = args.mode
        out_payload["_metadata"]["pool_size"] = len(kept)
    else:
        out_payload = kept
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_file).write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "input": args.in_file, "output": args.out_file, "mode": args.mode,
        "candidates_profile": args.candidate_profiles,
        "blocklist_file": args.blocklist_file,
        "blocklist_used": (Path(args.blocklist_file).exists() if args.blocklist_file else False),
        "n_input_cards": len(cards), "n_kept": len(kept),
        "n_rejected": len(rejected), "redaction_count": redaction_count,
        "pass_rate_pct": round(100.0 * len(kept) / max(1, len(cards)), 2),
        "top_foreign_names": [{"name": n, "reason": r, "count": c}
                              for (n, r), c in foreign_counter.most_common(50)],
        "rejected_card_ids": [r["id"] for r in rejected[:200]],
        "foreign_examples": foreign_examples[:30],
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("=" * 60)
    logger.info("Strict allowlist enforcement complete (v2.6.final)")
    logger.info("  Input cards         : %d", len(cards))
    logger.info("  Kept                : %d (%.2f%%)", len(kept), report["pass_rate_pct"])
    logger.info("  Rejected            : %d", len(rejected))
    if args.mode == "redact":
        logger.info("  Redactions made     : %d", redaction_count)
    if foreign_counter:
        logger.info("  Top foreign names   :")
        for (n, r), c in foreign_counter.most_common(5):
            logger.info("    - %r (%s) - %d occurrences", n, r, c)
    logger.info("=" * 60)
    logger.info("Output -> %s", args.out_file)
    logger.info("Report -> %s", args.report_out)


if __name__ == "__main__":
    main()

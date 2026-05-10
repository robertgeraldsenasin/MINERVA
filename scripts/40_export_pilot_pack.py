#!/usr/bin/env python3
"""
40_export_pilot_pack.py  (NEW v2.6-final, HANDOFF.md P1.2)
==========================================================

Builds a printable pre-pilot pack for SHS-student rater sessions.
Implements `HANDOFF.md` priority **P1.2** ("Pre-pilot with 5 SHS
students"). The output is a self-contained packet (one HTML file,
one questionnaire markdown, one answer-key markdown) that the user
can hand to a small classroom or pilot group without any further
preparation.

WHAT THIS DOES
--------------
1. Loads the curated card pool (output of script 24).
2. Samples N cards (default 30) stratified by:
     - verdict     (REAL / FAKE / UNCERTAIN, proportional to pool ratio)
     - tier        (novice / proficient / advanced, proportional)
     - candidate   (C-A / C-B / C-C, balanced ~ N/k each)
     - tactic      (greedy preference for un-seen tactics so that as
                    many of the 18 as possible appear in the pack)
   Sampling is deterministic given --seed (default 1729), so the
   pilot pack can be regenerated bit-for-bit at defense time.
3. Emits three files into --out_dir (default `reports/pilot_pack/`):
     - printable_card_pack.html
         A4 print CSS, one card per page, large readable Tagalog text,
         empty rater fields. No answers shown.
     - questionnaire.md
         The same five questions per card in markdown, suitable for
         copy-paste into a Google Form section.
     - answer_key.md
         The gold-truth scoring sheet: gold verdict, tactic, tier,
         fired DEPICT indicators, and the explanation-bank phrase
         that justifies the verdict.

THE FIVE QUESTIONS
------------------
  Q1. Is this post FAKE / REAL / UNCERTAIN?
  Q2. Why? (1-2 sentences, open response)
  Q3. Which manipulation tactic does it use? (single choice from the
      tactic list, plus "credible / not manipulation" for REAL cards)
  Q4. How confident are you? (1-5 scale)
  Q5. Would you share this post? (Yes / No / Maybe)

USAGE
-----
  python scripts/40_export_pilot_pack.py \\
      --pool_file generated/unity_cards_pool.json \\
      --out_dir reports/pilot_pack \\
      --n 30 --seed 1729

CITATIONS
---------
- Roozenbeek & van der Linden (2019). *Palgrave Communications 5*.
  Inoculation pre-/post-test rater protocol — basis for the per-card
  five-question schema.
- Modirrousta-Galian & Higham (2023). *Journal of Experimental
  Psychology: Applied*. Per-tier calibration — motivates the
  proportional tier sampling so the rater pack reflects the same
  difficulty mix the player will see.
- Hainmueller, Hangartner, & Yamamoto (2015). *PNAS 112(8)*.
  Vignette-experiment design — supports balanced candidate exposure
  (so rater opinions about a candidate do not leak into verdict
  judgements).
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Pool helpers
# --------------------------------------------------------------------- #
def load_pool(path: str | Path) -> list[dict]:
    """Read the curated pool JSON. Accepts either the bare list of
    cards or a wrapped {"_metadata": ..., "cards": [...]} payload.
    """
    payload = json.load(open(path, encoding="utf-8"))
    if isinstance(payload, dict) and "cards" in payload:
        return payload["cards"]
    return payload


def get_tactic(card: dict) -> str:
    return card.get("provenance", {}).get("tactic", "unknown")


def get_tier(card: dict) -> str:
    return card.get("provenance", {}).get("tier", "unknown")


# --------------------------------------------------------------------- #
# Stratified sampling
# --------------------------------------------------------------------- #
def _largest_remainder(counts: Counter, n: int) -> dict[str, int]:
    """Allocate `n` slots across keys proportional to `counts`,
    breaking ties via the largest-remainder method so the totals
    sum to exactly `n`.
    """
    total = sum(counts.values())
    if total == 0 or n == 0:
        return {k: 0 for k in counts}
    raw = {k: n * v / total for k, v in counts.items()}
    floors = {k: int(x) for k, x in raw.items()}
    remainders = sorted(
        ((raw[k] - floors[k], k) for k in raw),
        reverse=True,
    )
    deficit = n - sum(floors.values())
    out = dict(floors)
    for _, k in remainders[:deficit]:
        out[k] += 1
    return out


def stratified_sample(pool: list[dict], n: int = 30,
                      seed: int = 1729) -> list[dict]:
    """Greedy multi-criteria stratified sample.

    Phase 1: compute hard per-verdict quotas (largest-remainder so
    the quotas sum to exactly `n`).
    Phase 2: greedy fill — for every remaining card, score it by
    tactic novelty, candidate balance, and tier balance; pick the
    highest-scoring card whose verdict still has quota.

    Determinism: pool is shuffled with `random.Random(seed)`, and
    `max(..., key=...)` returns the first encountered maximum,
    so the same (pool, seed) always yields the same sample.
    """
    if n <= 0 or not pool:
        return []

    rng = random.Random(seed)
    pool_shuf = list(pool)
    rng.shuffle(pool_shuf)

    verdict_quota = _largest_remainder(
        Counter(c.get("verdict") for c in pool_shuf), n
    )
    tier_quota = _largest_remainder(
        Counter(get_tier(c) for c in pool_shuf), n
    )
    cand_codes = sorted({c.get("candidate") for c in pool_shuf
                         if c.get("candidate")})
    if cand_codes:
        per = n // len(cand_codes)
        extra = n % len(cand_codes)
        cand_quota = {c: per + (1 if i < extra else 0)
                      for i, c in enumerate(cand_codes)}
    else:
        cand_quota = {}

    selected: list[dict] = []
    used_ids: set[str] = set()
    used_tactics: set[str] = set()
    cand_count: Counter = Counter()
    tier_count: Counter = Counter()
    verdict_count: Counter = Counter()

    def score(card: dict) -> float:
        s = 0.0
        if get_tactic(card) not in used_tactics:
            s += 10.0
        cd = card.get("candidate")
        s -= 3.0 * max(0, cand_count[cd] - cand_quota.get(cd, 0))
        t = get_tier(card)
        s -= 2.0 * max(0, tier_count[t] - tier_quota.get(t, 0))
        return s

    while len(selected) < n:
        best = None
        best_score = float("-inf")
        for card in pool_shuf:
            if card.get("id") in used_ids:
                continue
            v = card.get("verdict")
            if verdict_count[v] >= verdict_quota.get(v, 0):
                continue
            sc = score(card)
            if sc > best_score:
                best_score = sc
                best = card

        if best is None:
            # All verdict buckets exhausted (small pools / odd quotas);
            # relax the verdict cap and pick any unseen card.
            for card in pool_shuf:
                if card.get("id") not in used_ids:
                    best = card
                    break
            if best is None:
                break

        selected.append(best)
        used_ids.add(best.get("id"))
        used_tactics.add(get_tactic(best))
        cand_count[best.get("candidate")] += 1
        tier_count[get_tier(best)] += 1
        verdict_count[best.get("verdict")] += 1

    return selected


# --------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------- #
def justifying_phrase(card: dict) -> str:
    """Phrase from the explanation bank that supports the gold verdict.

    For FAKE / UNCERTAIN cards: first non-CREDIBLE indicator phrase.
    For REAL cards: the CREDIBLE phrase if present, else summary.
    """
    expl = card.get("explanation", {}) or {}
    phrases = expl.get("indicator_phrases", []) or []
    if card.get("verdict") == "REAL":
        for p in phrases:
            if p.get("indicator") == "CREDIBLE":
                return p.get("phrase", "")
        return expl.get("summary", "") or ""
    for p in phrases:
        if p.get("indicator") and p.get("indicator") != "CREDIBLE":
            return p.get("phrase", "")
    return expl.get("summary", "") or ""


def write_html(cards: list[dict], out_path: str | Path) -> None:
    """Print-ready A4 card pack (one card per page, no answers)."""
    css = (
        "<style>\n"
        "  @page { size: A4; margin: 1.5cm; }\n"
        "  body { font-family: 'Georgia', 'Times New Roman', serif;\n"
        "         line-height: 1.55; color: #111; margin: 0; }\n"
        "  .card { page-break-after: always; padding: 1.5cm 1cm; }\n"
        "  .card:last-child { page-break-after: auto; }\n"
        "  .card-num { font-size: 10pt; color: #444; }\n"
        "  .card-id { font-family: 'Courier New', monospace;\n"
        "             font-size: 9pt; color: #777; margin-bottom: 1em; }\n"
        "  .post-text { font-size: 16pt; margin: 1.4em 0;\n"
        "               padding: 1em 1.2em; border-left: 4px solid #888;\n"
        "               background: #f7f7f7; }\n"
        "  .questions { margin-top: 1.2em; font-size: 12pt; }\n"
        "  .q { margin: 0.85em 0; }\n"
        "  .q-label { font-weight: bold; }\n"
        "  .checkbox { display: inline-block; width: 0.95em;\n"
        "              height: 0.95em; border: 1.5px solid #444;\n"
        "              vertical-align: middle; margin: 0 0.3em 0 0.6em; }\n"
        "  .write-line { display: block; border-bottom: 1px solid #888;\n"
        "                height: 1.4em; margin: 0.4em 0; }\n"
        "  h1 { font-size: 14pt; margin: 0 0 0.5em 0; }\n"
        "</style>"
    )

    parts: list[str] = [
        "<!doctype html>",
        "<html lang='tl'>",
        "<head>",
        "  <meta charset='utf-8'>",
        "  <title>MINERVA Pre-Pilot Card Pack</title>",
        css,
        "</head>",
        "<body>",
    ]

    total = len(cards)
    for i, c in enumerate(cards, 1):
        text = html.escape(c.get("text", ""))
        cid = html.escape(c.get("id", ""))
        parts.append("<section class='card'>")
        parts.append(f"  <h1>MINERVA Pre-Pilot</h1>")
        parts.append(f"  <div class='card-num'>Card {i} of {total}</div>")
        parts.append(f"  <div class='card-id'>ID: {cid}</div>")
        parts.append(f"  <div class='post-text'>{text}</div>")
        parts.append("  <div class='questions'>")
        parts.append(
            "    <div class='q'><span class='q-label'>Q1.</span> "
            "Is this post "
            "<span class='checkbox'></span>FAKE "
            "<span class='checkbox'></span>REAL "
            "<span class='checkbox'></span>UNCERTAIN ?</div>"
        )
        parts.append(
            "    <div class='q'><span class='q-label'>Q2.</span> "
            "Why? (1-2 sentences)"
            "<span class='write-line'></span>"
            "<span class='write-line'></span></div>"
        )
        parts.append(
            "    <div class='q'><span class='q-label'>Q3.</span> "
            "Which manipulation tactic does it use? "
            "(write the tactic name from the questionnaire)"
            "<span class='write-line'></span></div>"
        )
        parts.append(
            "    <div class='q'><span class='q-label'>Q4.</span> "
            "How confident are you? "
            "<span class='checkbox'></span>1 "
            "<span class='checkbox'></span>2 "
            "<span class='checkbox'></span>3 "
            "<span class='checkbox'></span>4 "
            "<span class='checkbox'></span>5</div>"
        )
        parts.append(
            "    <div class='q'><span class='q-label'>Q5.</span> "
            "Would you share this post? "
            "<span class='checkbox'></span>Yes "
            "<span class='checkbox'></span>No "
            "<span class='checkbox'></span>Maybe</div>"
        )
        parts.append("  </div>")
        parts.append("</section>")

    parts += ["</body>", "</html>", ""]
    Path(out_path).write_text("\n".join(parts), encoding="utf-8")


def write_questionnaire(cards: list[dict], tactic_options: list[str],
                        out_path: str | Path) -> None:
    """Five-question rater form per card, in markdown."""
    tactic_choices = ["credible / not manipulation"] + sorted(tactic_options)

    lines: list[str] = [
        "# MINERVA Pre-Pilot Questionnaire",
        "",
        "_Copy each block below into a single Google Form section "
        "(one section per card)._",
        "",
        f"**Total cards:** {len(cards)}",
        "",
        "---",
        "",
    ]

    for i, c in enumerate(cards, 1):
        lines.append(f"## Card {i} - `{c.get('id', '')}`")
        lines.append("")
        body = c.get("text", "").replace("\n", "\n> ")
        lines.append("> " + body)
        lines.append("")

        lines.append("**Q1. Is this post FAKE / REAL / UNCERTAIN?**")
        for opt in ("FAKE", "REAL", "UNCERTAIN"):
            lines.append(f"- [ ] {opt}")
        lines.append("")

        lines.append("**Q2. Why? (1-2 sentences, open response)**")
        lines.append("")
        lines.append("> _(rater's free-text answer here)_")
        lines.append("")

        lines.append(
            "**Q3. Which manipulation tactic does it use? (single choice)**"
        )
        for t in tactic_choices:
            lines.append(f"- [ ] {t}")
        lines.append("")

        lines.append(
            "**Q4. How confident are you? "
            "(1 = not at all, 5 = very confident)**"
        )
        for k in (1, 2, 3, 4, 5):
            lines.append(f"- [ ] {k}")
        lines.append("")

        lines.append("**Q5. Would you share this post?**")
        for opt in ("Yes", "No", "Maybe"):
            lines.append(f"- [ ] {opt}")
        lines.append("")

        lines.append("---")
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def write_answer_key(cards: list[dict], out_path: str | Path) -> None:
    """Scoring sheet: one block per card with gold verdict + indicators."""
    lines: list[str] = [
        "# MINERVA Pre-Pilot Answer Key",
        "",
        "_Use this to score completed questionnaires. Cards are listed "
        "in the same order as `questionnaire.md` and "
        "`printable_card_pack.html`._",
        "",
        f"**Total cards:** {len(cards)}",
        "",
        "---",
        "",
    ]

    for i, c in enumerate(cards, 1):
        verdict = c.get("verdict", "")
        tactic = get_tactic(c)
        tier = get_tier(c)
        cand = c.get("candidate", "")
        fired = c.get("fired_indicators", []) or []
        phrase = justifying_phrase(c)

        lines.append(f"## Card {i} - `{c.get('id', '')}`")
        lines.append("")
        lines.append(f"- **Gold verdict:** {verdict}")
        lines.append(f"- **Tactic:** `{tactic}`")
        lines.append(f"- **Tier:** {tier}")
        lines.append(f"- **Candidate:** {cand}")
        if fired:
            lines.append(
                f"- **Fired DEPICT indicators:** {', '.join(fired)}"
            )
        else:
            lines.append("- **Fired DEPICT indicators:** (none)")
        lines.append(f"- **Justifying phrase:** {phrase}")
        lines.append("")
        lines.append("---")
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build a 30-card pre-pilot pack for SHS-student raters "
            "(HANDOFF.md P1.2)."
        )
    )
    p.add_argument(
        "--pool_file",
        default="generated/unity_cards_pool.json",
        help="Curated pool JSON (output of script 24)",
    )
    p.add_argument(
        "--out_dir",
        default="reports/pilot_pack",
        help="Directory for the three output files",
    )
    p.add_argument(
        "--n", type=int, default=30,
        help="Number of cards to sample (default: 30)",
    )
    p.add_argument(
        "--seed", type=int, default=1729,
        help="RNG seed for reproducible sampling (default: 1729)",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    pool = load_pool(args.pool_file)
    logger.info("Loaded %d cards from %s", len(pool), args.pool_file)

    sample = stratified_sample(pool, n=args.n, seed=args.seed)
    tactic_options = sorted({get_tactic(c) for c in pool})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / "printable_card_pack.html"
    quest_path = out_dir / "questionnaire.md"
    key_path = out_dir / "answer_key.md"

    write_html(sample, html_path)
    write_questionnaire(sample, tactic_options, quest_path)
    write_answer_key(sample, key_path)

    verdicts = Counter(c.get("verdict") for c in sample)
    tiers = Counter(get_tier(c) for c in sample)
    cands = Counter(c.get("candidate") for c in sample)
    sampled_tactics = sorted({get_tactic(c) for c in sample})

    logger.info("=" * 60)
    logger.info("Pre-pilot pack built (HANDOFF.md P1.2)")
    logger.info("  Sampled cards     : %d", len(sample))
    logger.info("  Verdict dist      : %s", dict(verdicts))
    logger.info("  Tier dist         : %s", dict(tiers))
    logger.info("  Candidate dist    : %s", dict(cands))
    logger.info(
        "  Tactic coverage   : %d / %d  (sampled: %s)",
        len(sampled_tactics), len(tactic_options),
        ", ".join(sampled_tactics),
    )
    logger.info("  Seed              : %d", args.seed)
    logger.info("  Generated at      : %s",
                datetime.now(timezone.utc).isoformat())
    logger.info("=" * 60)
    logger.info("HTML pack         -> %s", html_path)
    logger.info("Questionnaire     -> %s", quest_path)
    logger.info("Answer key        -> %s", key_path)


if __name__ == "__main__":
    main()

"""
minerva_response_bank.py
========================

Indicator-fired, tiered response bank that replaces the static
"Verdict: REAL/FAKE … The decision comes from the stored Qlattice
equation applied to detector/embedding features." boilerplate of
the legacy pipeline.

Design rationale
----------------
* Each indicator (12 total — see minerva_indicators.py) has multiple
  phrasing variants per tier (novice / proficient / advanced).
* Selection is DETERMINISTIC per card via stable hash so the SAME
  card always gets the SAME explanation, but the COHORT-WIDE
  distribution is varied — preserving faithfulness (Longo et al.
  2024 Open Problem 7) while delivering pedagogically meaningful
  variety (Dehghanzadeh et al. 2024).
* Each variant ends with a SIFT move (Caulfield 2019;
  Caulfield & Wineburg 2023): Stop / Investigate / Find / Trace.
* Credible cards get explicit POSITIVE feedback, addressing the
  conservative-response-bias risk identified by Modirrousta-Galian
  & Higham (2023).
* Bank is versioned (bank_version = "1.0") and the hash is stamped
  in every card's provenance for A/B comparability.

Citations: Barzilai & Stadtler (2025); Roozenbeek & van der Linden
(2019); Khosravi et al. (2022); Athira et al. (2023); Liu, Ye & Li
(2024); Bautista (2021).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

BANK_VERSION = "1.1"


# ---------------------------------------------------------------------------
# SIFT moves (Caulfield 2019)
# ---------------------------------------------------------------------------
SIFT_MOVES = {
    "stop":       "Stop and breathe before sharing.",
    "investigate": "Investigate the source — who is publishing this?",
    "find":       "Find better coverage from a known outlet.",
    "trace":      "Trace claims and quotes back to the original document.",
}


# ---------------------------------------------------------------------------
# Bank entries
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BankEntry:
    code: str          # indicator code
    tier: str          # novice / proficient / advanced
    phrase: str        # the actual feedback line
    sift: str          # one of the 4 SIFT moves
    bank_id: str       # stable id for provenance, e.g. "EMO/v1/n1"


def _e(code: str, tier: str, idx: int, phrase: str, sift: str) -> BankEntry:
    """Compact constructor."""
    bank_id = f"{code}/v{BANK_VERSION}/{tier[0]}{idx}"
    return BankEntry(code=code, tier=tier, phrase=phrase, sift=sift,
                     bank_id=bank_id)


# ---------------------------------------------------------------------------
# The bank — 12 indicators × 3 tiers × 2-3 variants ≈ 78 entries
# ---------------------------------------------------------------------------
BANK: dict[str, list[BankEntry]] = {
    # ============================== EMO ==============================
    "EMO": [
        _e("EMO", "novice", 1,
           "Notice the heated words like *betrayed* and *outraged*. Strong "
           "feelings are a classic emotion trick — they push you to react "
           "before you check.",
           SIFT_MOVES["stop"]),
        _e("EMO", "novice", 2,
           "This post leads with anger before facts. Posts that lead with "
           "feeling often hide weak evidence.",
           SIFT_MOVES["find"]),
        _e("EMO", "novice", 3,
           "Words like *scandalo* and *manloloko* are designed to make you "
           "share without thinking. Pause first.",
           SIFT_MOVES["stop"]),
        _e("EMO", "proficient", 1,
           "The emotional-word density here is well above what news desks "
           "typically use. That's a manipulation cue, not a truth cue — it "
           "signals the writer wanted reactions, not readers.",
           SIFT_MOVES["investigate"]),
        _e("EMO", "proficient", 2,
           "This is the **Emotion** technique from the Bad News taxonomy "
           "(Roozenbeek & van der Linden). Once you name the technique, "
           "it loses some of its grip.",
           SIFT_MOVES["investigate"]),
        _e("EMO", "advanced", 1,
           "Sensation-axis intent reads high here while credibility-axis "
           "intent reads low. The post is engineered for shareability, not "
           "verifiability. Compare it with the candidate's official "
           "statement on the same topic.",
           SIFT_MOVES["trace"]),
    ],

    # ============================== URG ==============================
    "URG": [
        _e("URG", "novice", 1,
           "ALL CAPS and *SHARE NOW* are pressure tactics. Real news doesn't "
           "beg you to reshare in 10 minutes.",
           SIFT_MOVES["stop"]),
        _e("URG", "novice", 2,
           "When a post tells you to act *bago mahuli* or *before it's "
           "deleted*, it's training you to skip checking.",
           SIFT_MOVES["trace"]),
        _e("URG", "novice", 3,
           "Notice the urgency: *agad-agad*, *ngayon na*, three exclamation "
           "marks. Urgency cues are designed to bypass your judgement.",
           SIFT_MOVES["stop"]),
        _e("URG", "proficient", 1,
           "Urgency framing — *before they take this down* — is a textbook "
           "Emotion+Trolling combo from the DEPICT taxonomy. It pairs panic "
           "with social pressure.",
           SIFT_MOVES["stop"]),
        _e("URG", "proficient", 2,
           "Two convergent urgency cues here (shouting capitals AND a "
           "*share-before-deleted* phrase). Stack matters: each cue alone "
           "could be coincidence; together they are a pattern.",
           SIFT_MOVES["investigate"]),
        _e("URG", "advanced", 1,
           "The urgency-density score is in the top 10% of the training "
           "distribution for fake-labelled posts. This is a high-precision "
           "signal in our QLattice rule set.",
           SIFT_MOVES["stop"]),
    ],

    # ============================== ANON =============================
    "ANON": [
        _e("ANON", "novice", 1,
           "*'Sources say'* with no name is not a source. Real reporters "
           "name their sources, or explain why they cannot.",
           SIFT_MOVES["investigate"]),
        _e("ANON", "novice", 2,
           "Watch for *'insiders revealed'* with nobody named. If no one "
           "will sign their name to a claim, it isn't reportable yet.",
           SIFT_MOVES["investigate"]),
        _e("ANON", "novice", 3,
           "Tagalog hearsay markers like *daw* and *diumano* signal "
           "second-hand information. Treat as rumour until traced.",
           SIFT_MOVES["trace"]),
        _e("ANON", "proficient", 1,
           "Anonymous-source phrasing without a follow-up named source is a "
           "credibility red flag in the W3C credibility-signals catalogue. "
           "Real anonymous sources always pair with a named verifier.",
           SIFT_MOVES["investigate"]),
        _e("ANON", "advanced", 1,
           "The post uses anonymous attribution as its only evidence anchor. "
           "Cross-reference: does any *named* outlet report the same claim "
           "with the same details within 24 hours? If not, treat as a single "
           "unverified rumour.",
           SIFT_MOVES["find"]),
    ],

    # ============================== MISS =============================
    "MISS": [
        _e("MISS", "novice", 1,
           "There's no link, no document, no photo with caption — the claim "
           "hangs in the air. Big claims need receipts.",
           SIFT_MOVES["find"]),
        _e("MISS", "novice", 2,
           "Big claim, zero receipts. Ask yourself: where would I check "
           "this? If you cannot answer that question, the post hasn't done "
           "its job.",
           SIFT_MOVES["trace"]),
        _e("MISS", "proficient", 1,
           "Notice the asymmetry: a bold political claim with no document, "
           "contract, or COA report attached. The bigger the claim, the "
           "bigger the receipts should be.",
           SIFT_MOVES["find"]),
        _e("MISS", "advanced", 1,
           "On the credibility-signals axis (Leite et al. 2025), this post "
           "scores zero on both *named-source* and *external-link* signals. "
           "These are two of the 23 W3C indicators; both absent is a strong "
           "low-credibility prior.",
           SIFT_MOVES["find"]),
    ],

    # ============================== FAB ==============================
    "FAB": [
        _e("FAB", "novice", 1,
           "A direct quote from a public figure with no video, transcript, "
           "or news link is suspicious — anyone could write quote marks "
           "around anything.",
           SIFT_MOVES["trace"]),
        _e("FAB", "novice", 2,
           "Long quotes attributed to politicians should always come with a "
           "transcript or video. No transcript = treat as alleged.",
           SIFT_MOVES["trace"]),
        _e("FAB", "proficient", 1,
           "Fabricated-quote attacks were a 2022 Philippine-election staple "
           "(Arugay & Baquisal 2022). Always trace to a primary "
           "transcript before treating a quote as real.",
           SIFT_MOVES["trace"]),
        _e("FAB", "advanced", 1,
           "Quote-fabrication is a sub-pattern of Impersonation in the "
           "DEPICT taxonomy. It works because readers trust quotation "
           "marks. The defence is mechanical: trace to source.",
           SIFT_MOVES["trace"]),
    ],

    # ============================== POL ==============================
    "POL": [
        _e("POL", "novice", 1,
           "*Real Filipinos vs traitors* — that's an us-vs-them frame. "
           "Real reporting describes people, not enemies.",
           SIFT_MOVES["find"]),
        _e("POL", "novice", 2,
           "Notice the *us vs them* labels. These frames make the other "
           "side feel less than human, which is the point — and the trap.",
           SIFT_MOVES["stop"]),
        _e("POL", "proficient", 1,
           "Polarising framing — Bad News *Polarization* technique — "
           "trades nuance for tribal certainty. Notice it; resist the pull "
           "even when it flatters your side.",
           SIFT_MOVES["investigate"]),
        _e("POL", "advanced", 1,
           "Polarising frames have a recognisable shape: in-group purity, "
           "out-group treachery, no middle ground. Once you can name the "
           "shape, you can step outside it without changing your mind on "
           "the underlying issue.",
           SIFT_MOVES["investigate"]),
    ],

    # ============================== CONS =============================
    "CONS": [
        _e("CONS", "novice", 1,
           "*'They don't want you to know'* — this builds a secret-cabal "
           "story. Secrets that big leave footprints; check if anyone has "
           "found them.",
           SIFT_MOVES["investigate"]),
        _e("CONS", "novice", 2,
           "Conspiratorial framing relies on you *not* checking. The fix "
           "is the simplest one: check.",
           SIFT_MOVES["find"]),
        _e("CONS", "proficient", 1,
           "The Conspiracy technique (DEPICT) is recognisable by an "
           "appeal to hidden actors and unfalsifiable claims. *No matter "
           "what evidence we find, the cover-up is bigger* — that's the "
           "shape of the trap.",
           SIFT_MOVES["investigate"]),
        _e("CONS", "advanced", 1,
           "Conspiracy posts often fuse with Discrediting: the journalist "
           "or fact-checker is *part of the cover-up*. Track which "
           "credible sources the post tries to pre-emptively discredit; "
           "those are usually the ones to check.",
           SIFT_MOVES["find"]),
    ],

    # ============================== DISC =============================
    "DISC": [
        _e("DISC", "novice", 1,
           "This attacks the person, not the policy. Even if the insult "
           "is funny, it isn't evidence.",
           SIFT_MOVES["find"]),
        _e("DISC", "novice", 2,
           "Notice the personal attack. Ask: would I find this convincing "
           "if it were aimed at someone I support? If not, set it aside.",
           SIFT_MOVES["stop"]),
        _e("DISC", "proficient", 1,
           "Red-tagging — labelling a candidate or critic as *communist* "
           "or *NPA-supporter* without evidence — is a documented "
           "Philippine disinformation tactic (Schipper 2025). Treat it as "
           "a signal of a weak factual case, not a strong one.",
           SIFT_MOVES["investigate"]),
        _e("DISC", "advanced", 1,
           "Discrediting is a load-bearing technique in Philippine "
           "electoral disinformation (Arugay & Baquisal 2022). It works "
           "because it short-circuits the evidence question by replacing "
           "*Is this true?* with *Are they trustworthy?*. Notice the "
           "switch; refuse it.",
           SIFT_MOVES["investigate"]),
    ],

    # ============================== IMP ==============================
    "IMP": [
        _e("IMP", "novice", 1,
           "The logo looks like a real news brand, but the URL is off by "
           "a letter or the domain is wrong. That's a copy-cat page.",
           SIFT_MOVES["investigate"]),
        _e("IMP", "novice", 2,
           "Spoofed outlet names are easy to make and easy to spot once "
           "you know to look. Check the domain against the real outlet's "
           "homepage.",
           SIFT_MOVES["trace"]),
        _e("IMP", "proficient", 1,
           "Impersonation pages mimic a known outlet's brand to borrow "
           "its credibility. The defence is one click: search the real "
           "outlet's website for the headline.",
           SIFT_MOVES["find"]),
    ],

    # ============================== REV ==============================
    "REV": [
        _e("REV", "novice", 1,
           "Claims about a *golden age* that contradict textbooks and "
           "documented history are a known Philippine disinformation "
           "pattern.",
           SIFT_MOVES["find"]),
        _e("REV", "novice", 2,
           "If a post rewrites a historical period as much better than "
           "documented archives say, ask: who benefits from that "
           "rewriting?",
           SIFT_MOVES["investigate"]),
        _e("REV", "proficient", 1,
           "Historical revisionism was the *dominant* misinformation "
           "narrative of the 2022 Philippine elections (Arugay & "
           "Baquisal 2022). Spotting it is a core voter-literacy skill.",
           SIFT_MOVES["find"]),
        _e("REV", "advanced", 1,
           "Revisionism rests on three moves: erase abuses, inflate "
           "achievements, and discredit historians. If you see two of "
           "the three in one post, you are looking at the pattern, not "
           "an honest disagreement about history.",
           SIFT_MOVES["find"]),
    ],

    # ============================== ENDO =============================
    "ENDO": [
        _e("ENDO", "novice", 1,
           "*'85% of Filipinos already support…'* — but who counted? "
           "Real surveys disclose method, sample size, and dates.",
           SIFT_MOVES["trace"]),
        _e("ENDO", "novice", 2,
           "A poll without a polling firm, sample size, or date is not "
           "a poll — it's a graphic. Treat it as opinion, not data.",
           SIFT_MOVES["investigate"]),
        _e("ENDO", "proficient", 1,
           "Manufactured-survey claims usually omit four things: firm "
           "name, sample, dates, and methodology. The omission is the "
           "evidence — real polls publish all four.",
           SIFT_MOVES["trace"]),
        _e("ENDO", "advanced", 1,
           "Cross-reference against the SWS, Pulse Asia, OCTA, Stratbase, "
           "Laylo, or Publicus archives. Polls that exist in the wild "
           "but not in any reputable firm's records are graphics, not "
           "data.",
           SIFT_MOVES["find"]),
    ],

    # ============================== RECF =============================
    "RECF": [
        _e("RECF", "novice", 1,
           "An invented project or fake award attached to a candidate "
           "is record fabrication. Check the candidate's official site "
           "or COA.",
           SIFT_MOVES["trace"]),
        _e("RECF", "novice", 2,
           "Big credentials (Harvard, Nobel, UN award) attached to a "
           "candidate without a verifiable record are usually invented.",
           SIFT_MOVES["investigate"]),
        _e("RECF", "proficient", 1,
           "Cross-reference the claim against the candidate's verified "
           "Senate or House voting record. Inventions don't survive "
           "cross-referencing — that's why fact-checkers can debunk them "
           "in minutes.",
           SIFT_MOVES["find"]),
        _e("RECF", "advanced", 1,
           "Record fabrication and manufactured endorsement often appear "
           "together: the invented credential is paired with an invented "
           "endorser. Both fail the same test — provenance.",
           SIFT_MOVES["trace"]),
    ],
}


# ---------------------------------------------------------------------------
# Credible-card affirmations (Modirrousta-Galian & Higham 2023 mandate)
# v2.2: phrasing rewritten to NOT make claims that the indicator extractor
# may not actually verify (e.g. "verifiable date", "named outlet"). Instead
# focus on the absence of misinformation cues — which is what we can
# actually attest to from rule-based detection.
# ---------------------------------------------------------------------------
CREDIBLE_AFFIRMATIONS = [
    BankEntry(code="CREDIBLE", tier="novice",
              phrase="Good news — this post does not raise any of our 12 "
                     "common misinformation flags. That low-noise signal is "
                     "what credible content tends to look like.",
              sift=SIFT_MOVES["find"], bank_id="CREDIBLE/v1.1/n1"),
    BankEntry(code="CREDIBLE", tier="novice",
              phrase="None of the standard misinformation cues fired here — "
                     "no urgency push, no anonymous attribution, no loaded "
                     "wording. Reward yourself for noticing the calm.",
              sift=SIFT_MOVES["find"], bank_id="CREDIBLE/v1.1/n2"),
    BankEntry(code="CREDIBLE", tier="proficient",
              phrase="Zero misinformation indicators triggered. Use this as a "
                     "baseline: when several flags fire on a real post in "
                     "your feed, contrast against the absence-of-flags "
                     "pattern you see here.",
              sift=SIFT_MOVES["find"], bank_id="CREDIBLE/v1.1/p1"),
    BankEntry(code="CREDIBLE", tier="advanced",
              phrase="Posts like this are the antidote to the conservative-bias "
                     "trap (Modirrousta-Galian & Higham 2023): they remind you "
                     "that *trust*, calibrated to evidence, is a real-life "
                     "skill — not just *suspicion*.",
              sift=SIFT_MOVES["find"], bank_id="CREDIBLE/v1.1/a1"),
]


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------
def _stable_index(seed_str: str, n: int) -> int:
    h = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(n, 1)


def select_for_indicator(
    indicator_code: str,
    *,
    tier: str,
    seed_str: str,
) -> BankEntry | None:
    """Deterministic selection: same (indicator, tier, seed) -> same phrase."""
    entries = [e for e in BANK.get(indicator_code, []) if e.tier == tier]
    if not entries:
        # Fallback: try novice tier
        entries = [e for e in BANK.get(indicator_code, []) if e.tier == "novice"]
    if not entries:
        return None
    return entries[_stable_index(seed_str + indicator_code + tier, len(entries))]


def select_credible(
    *,
    tier: str,
    seed_str: str,
) -> BankEntry:
    """Pick a credible-card affirmation."""
    entries = [e for e in CREDIBLE_AFFIRMATIONS if e.tier == tier]
    if not entries:
        entries = CREDIBLE_AFFIRMATIONS
    return entries[_stable_index(seed_str + "CREDIBLE" + tier, len(entries))]


def tier_for_card_index(idx_within_session: int) -> str:
    """Difficulty banding: tier graduates as the player advances."""
    if idx_within_session < 10:
        return "novice"
    if idx_within_session < 25:
        return "proficient"
    return "advanced"


def assemble_explanation(
    *,
    fired_indicators: list[str],
    verdict: str,                # "FAKE" | "REAL" | "UNCERTAIN"
    fake_likelihood_percent: float,
    seed_str: str,
    tier: str = "novice",
    candidate_name: str | None = None,
) -> dict:
    """
    Build a full ExplanationBlock-compatible dict.

    The structure is content-aware: phrases are drawn ONLY for the
    indicators that actually fired, the SIFT move is taken from the
    most-prominent indicator's bank entry, and the summary is composed
    from those building blocks.
    """
    if verdict == "REAL":
        cred = select_credible(tier=tier, seed_str=seed_str)
        phrases = [{
            "indicator": "CREDIBLE",
            "phrase": cred.phrase,
            "bank_ref": cred.bank_id,
            "sift_move": cred.sift,
        }]
        summary = (
            f"This post looks credible (estimated fake-likelihood "
            f"{fake_likelihood_percent:.0f}%). " + cred.phrase
        )
        return {
            "tier": tier,
            "summary": summary,
            "indicator_phrases": phrases,
            "sift_move": cred.sift,
            "credible_counter_card_id": None,
            "bank_version": BANK_VERSION,
        }

    # FAKE / UNCERTAIN: build from fired indicators
    if not fired_indicators:
        # No indicators fired but verdict is FAKE — surface a generic prompt
        # that still names the verdict and pushes a SIFT move.
        cand_phrase = (
            f" The post mentions {candidate_name}." if candidate_name else ""
        )
        summary = (
            f"This post looks suspicious (estimated fake-likelihood "
            f"{fake_likelihood_percent:.0f}%), but the specific cues are "
            f"subtle.{cand_phrase} {SIFT_MOVES['investigate']}"
        )
        return {
            "tier": tier,
            "summary": summary,
            "indicator_phrases": [],
            "sift_move": SIFT_MOVES["investigate"],
            "credible_counter_card_id": None,
            "bank_version": BANK_VERSION,
        }

    # Order indicators by a fixed priority so the most-teachable cue
    # leads (rather than alphabetical accident).
    PRIORITY = ["EMO", "URG", "ANON", "MISS", "FAB", "REV", "ENDO",
                "RECF", "POL", "CONS", "DISC", "IMP"]
    ordered = sorted(
        fired_indicators,
        key=lambda c: PRIORITY.index(c) if c in PRIORITY else 99,
    )

    phrases_out = []
    for code in ordered[:3]:  # cap at 3 for cognitive load
        entry = select_for_indicator(code, tier=tier, seed_str=seed_str)
        if entry is None:
            continue
        phrases_out.append({
            "indicator": code,
            "phrase": entry.phrase,
            "bank_ref": entry.bank_id,
            "sift_move": entry.sift,
        })

    if not phrases_out:
        # Defensive: no bank entries matched — fall back to subtle prompt
        return assemble_explanation(
            fired_indicators=[],
            verdict=verdict,
            fake_likelihood_percent=fake_likelihood_percent,
            seed_str=seed_str,
            tier=tier,
            candidate_name=candidate_name,
        )

    cand_intro = f"This post about {candidate_name}" if candidate_name else "This post"
    lead = (
        f"{cand_intro} looks suspicious (estimated fake-likelihood "
        f"{fake_likelihood_percent:.0f}%). "
        f"{len(phrases_out)} misinformation cue"
        f"{'s' if len(phrases_out) != 1 else ''} fired: "
        f"{', '.join(p['indicator'] for p in phrases_out)}."
    )
    body = " ".join(p["phrase"] for p in phrases_out)
    sift = phrases_out[0]["sift_move"]
    summary = f"{lead} {body}"

    return {
        "tier": tier,
        "summary": summary,
        "indicator_phrases": phrases_out,
        "sift_move": sift,
        "credible_counter_card_id": None,
        "bank_version": BANK_VERSION,
    }


def bank_hash() -> str:
    """Stable hash of the bank contents for provenance."""
    payload = []
    for code, entries in sorted(BANK.items()):
        for e in entries:
            payload.append(f"{e.bank_id}|{e.phrase}|{e.sift}")
    for e in CREDIBLE_AFFIRMATIONS:
        payload.append(f"{e.bank_id}|{e.phrase}|{e.sift}")
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()[:16]


def bank_stats() -> dict:
    """Reporting stats."""
    by_tier: dict[str, int] = {"novice": 0, "proficient": 0, "advanced": 0}
    total = 0
    for entries in BANK.values():
        for e in entries:
            by_tier[e.tier] += 1
            total += 1
    for e in CREDIBLE_AFFIRMATIONS:
        by_tier[e.tier] += 1
        total += 1
    return {
        "bank_version": BANK_VERSION,
        "bank_hash": bank_hash(),
        "total_entries": total,
        "indicators_covered": len(BANK),
        "by_tier": by_tier,
        "credible_affirmations": len(CREDIBLE_AFFIRMATIONS),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Bank stats:", bank_stats())
    for tier in ("novice", "proficient", "advanced"):
        print(f"\n=== Sample FAKE explanation, tier={tier} ===")
        out = assemble_explanation(
            fired_indicators=["EMO", "URG", "ANON", "MISS"],
            verdict="FAKE",
            fake_likelihood_percent=83.4,
            seed_str="card_demo_001",
            tier=tier,
            candidate_name="Sen. Reynaldo \"Rey\" Marquez",
        )
        print(out["summary"][:600])
    print("\n=== Sample REAL explanation ===")
    out = assemble_explanation(
        fired_indicators=[],
        verdict="REAL",
        fake_likelihood_percent=4.1,
        seed_str="card_demo_002",
        tier="novice",
    )
    print(out["summary"])


from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_PROFILES = {
    "GENERAL": {
        "name": "the San Isidro election process",
        "role": "general election context",
        "aliases": ["city election", "election bulletin", "official notice"],
        "platform_keywords": [
            "debate", "election notice", "bulletin", "city website", "official schedule",
            "official election bulletin", "city notice", "campaign schedule"
        ],
        "trusted_sources": [
            "official election bulletin", "city website", "city election office"
        ],
    },
    "A": {
        "name": "Aurelia Santos",
        "role": "incumbent mayor",
        "aliases": ["Candidate A", "Aurelia Santos", "Mayor Santos", "Santos"],
        "platform_keywords": [
            "public health", "clinic", "ordinance", "barangay", "town hall",
            "health expansion", "crime rate", "construction firm", "public filings",
            "city report", "city council archive"
        ],
        "trusted_sources": [
            "official city website", "city council archive", "barangay attendance records",
            "city annual report", "public filings"
        ],
    },
    "B": {
        "name": "Bruno Villanueva",
        "role": "mayoral challenger",
        "aliases": ["Candidate B", "Bruno Villanueva", "Villanueva", "Bruno"],
        "platform_keywords": [
            "student transit", "transit subsidy", "city council approval", "open data",
            "procurement summaries", "proposal costings", "campaign website",
            "quarterly report", "costings", "proposal"
        ],
        "trusted_sources": [
            "official campaign website", "city council minutes",
            "budget office memo", "procurement summary page"
        ],
    },
    "C": {
        "name": "Celia Navarro",
        "role": "law-and-order challenger",
        "aliases": ["Candidate C", "Celia Navarro", "Navarro", "Celia"],
        "platform_keywords": [
            "law and order", "crime statistics", "city-specific records",
            "budget plan", "line-item budget", "procurement transparency",
            "campaign speech", "crime records", "police", "budget"
        ],
        "trusted_sources": [
            "official campaign website", "city police report",
            "city budget office page", "debate transcript"
        ],
    },
}

DEFAULT_BANK = {
    "frameworks": {
        "civic_online_reasoning": [
            "Who is behind this information?",
            "What is the evidence?",
            "What do other sources say?",
        ],
        "sift": [
            "Stop before reacting or sharing.",
            "Investigate the source or uploader.",
            "Find better coverage from reliable outlets.",
            "Trace the quote, image, number, or clip to its original context.",
        ],
    },
    "tactics": {
        "unverified_quote": {
            "title": "Unverified quote",
            "why_it_matters": "A quote can be edited, invented, or pulled out of context.",
            "default_steps": [
                "Search the exact quote plus the candidate name.",
                "Look for a full speech, transcript, press release, or debate clip.",
                "Check whether credible outlets quote the same words in the same context.",
            ],
        },
        "unsupported_number": {
            "title": "Unsupported number",
            "why_it_matters": "Percentages, rankings, and totals sound persuasive even when the source is missing or misleading.",
            "default_steps": [
                "Identify the number being used to persuade you.",
                "Ask which document, survey, or report produced that number.",
                "Compare it with the official city or campaign source.",
            ],
        },
        "contextless_media": {
            "title": "Contextless media",
            "why_it_matters": "Photos, screenshots, and short clips often hide where, when, and why something happened.",
            "default_steps": [
                "Find the original upload, not just a repost or edited clip.",
                "Check the date, location, and the full sequence of events.",
                "See whether reputable coverage matches the same interpretation.",
            ],
        },
        "false_endorsement": {
            "title": "False endorsement or affiliation",
            "why_it_matters": "Campaign influence can be exaggerated by claiming support that has no official record.",
            "default_steps": [
                "Check whether the endorsement appears in official filings or statements.",
                "Look for the endorser's own verified page or press release.",
                "Compare the claim with public records and campaign disclosures.",
            ],
        },
        "smear_or_accusation": {
            "title": "Smear or accusation",
            "why_it_matters": "Serious allegations trigger emotion quickly, but they still need traceable evidence and context.",
            "default_steps": [
                "Separate the accusation from the evidence.",
                "Look for a police report, court document, or official statement.",
                "Check whether credible outlets report the same facts without exaggeration.",
            ],
        },
        "policy_distortion": {
            "title": "Policy distortion",
            "why_it_matters": "Posts often twist a real proposal by changing timing, cost, scope, or approval status.",
            "default_steps": [
                "Identify the actual policy or proposal being discussed.",
                "Open the official program page, ordinance, or campaign costing.",
                "Compare the post's wording with the source document line by line.",
            ],
        },
        "virality_without_evidence": {
            "title": "Virality used as evidence",
            "why_it_matters": "A claim does not become true just because many people react to it.",
            "default_steps": [
                "Ignore likes and shares for the moment.",
                "Trace the claim to the earliest traceable source.",
                "Check whether the source provides evidence, not just reactions.",
            ],
        },
        "traceable_update": {
            "title": "Traceable update",
            "why_it_matters": "A credible post still needs verification habits; trust should come from evidence, not from tone alone.",
            "default_steps": [
                "Identify the source named in the post.",
                "Verify that the source is official or independently credible.",
                "Check whether another reliable source reports the same update.",
            ],
        },
        "ambiguous_or_uncertain": {
            "title": "Ambiguous claim",
            "why_it_matters": "Some posts mix a real issue with weak or missing evidence, so the best response is caution and verification.",
            "default_steps": [
                "List what the post clearly shows and what it only implies.",
                "Find the missing evidence that would settle the unclear part.",
                "Delay trusting or sharing until you confirm the unclear detail.",
            ],
        },
    },
}

GENERAL_ELECTION_TERMS = [
    "candidate", "kandidato", "campaign", "kampanya", "vote", "boto",
    "election", "eleksyon", "debate", "rally", "survey", "poll", "platform",
    "plataporma", "mayor", "council", "city", "barangay", "official", "website",
    "archive", "proposal", "ordinance", "budget", "town hall", "councilor",
    "endorsement", "procurement", "report", "bulletin", "transparency"
]

SOURCE_PATTERNS = [
    r"\bayon sa\b", r"\baccording to\b", r"\bofficial\b", r"\bwebsite\b", r"\barchive\b",
    r"\brecords?\b", r"\breport\b", r"\btranscript\b", r"\bbulletin\b",
    r"\bcity council\b", r"\bcommission\b", r"\bposted\b", r"\bpublished\b",
    r"\bpublic filings\b", r"\bminutes\b"
]

EMOTION_PATTERNS = [
    r"\bbreaking\b", r"\btrending\b", r"\bbabala\b", r"\bpaalala\b", r"\bviral\b",
    r"\bpakishare\b", r"\bshare now\b", r"\burgent\b", r"\bshock\b", r"\bscandal\b"
]

QUOTE_PATTERNS = [
    r"\".+?\"", r"\bsabi ni\b", r"\baniya\b", r"\bpahayag\b", r"\bsaid\b",
]

NUMBER_PATTERNS = [
    r"\b\d+(?:\.\d+)?%\b", r"\b\d{1,3}(?:,\d{3})+\b", r"\b\d+\s*(?:million|billion|thousand)\b",
    r"\b\d+\b"
]

OFFTOPIC_PATTERNS = {
    "sports": r"\b(WBO|NBA|PBA|semis|rebounds|blocks|quarter|coach|athlete|gold medal|career-high|Grandmaster|kampeon|best-of-seven|finals)\b",
    "showbiz": r"\b(movie|music|beauty queen|fan|tagahanga|love team|lovelife)\b",
    "transport_util": r"\b(Grab|Meralco|transport strike|fare increase|kuryente)\b",
    "violent_incident": r"\b(pinagbabaril|pamamaril|fraternity|suspek|suspect|shooting|salarin)\b",
}

MALFORMED_ENDING_RE = re.compile(r"(?:\b(?:happy|said|this|will|that|we|safe)\')$|\.{3}$", re.I)


def stable_index(seed: str, size: int) -> int:
    if size <= 0:
        return 0
    h = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % size


def choose(seed: str, items: Sequence[str]) -> str:
    return items[stable_index(seed, len(items))]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_profiles(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return dict(DEFAULT_PROFILES)
    payload = read_json(path)
    if isinstance(payload, dict) and "candidates" in payload:
        out = {}
        for item in payload["candidates"]:
            cid = str(item.get("candidate_id") or item.get("id") or item.get("code"))
            if cid:
                out[cid] = dict(item)
        for k, v in payload.items():
            if k != "candidates" and isinstance(v, dict):
                out[str(k)] = dict(v)
        return out
    if isinstance(payload, list):
        out = {}
        for item in payload:
            cid = str(item.get("candidate_id") or item.get("id") or item.get("code"))
            if cid:
                out[cid] = dict(item)
        return out
    if isinstance(payload, dict):
        return {str(k): dict(v) for k, v in payload.items()}
    raise ValueError("Unsupported candidate profile JSON structure.")


def load_bank(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_BANK))
    return read_json(path)


def normalize_target(card: Dict[str, Any]) -> str:
    for key in ("targets", "target", "candidate", "candidate_id"):
        if key in card and card[key]:
            val = card[key]
            if isinstance(val, list):
                return str(val[0])
            return str(val)
    classification = card.get("classification")
    if isinstance(classification, dict):
        val = classification.get("targets")
        if isinstance(val, list) and val:
            return str(val[0])
    linked = card.get("linked_blue_truth", {})
    if isinstance(linked, dict) and linked.get("candidate"):
        return str(linked["candidate"])
    return "GENERAL"


def extract_blue_truth_keywords(card: Dict[str, Any]) -> List[str]:
    bt = card.get("linked_blue_truth", {})
    if not isinstance(bt, dict):
        return []
    text = str(bt.get("text", "") or "").lower()
    tokens = [t for t in re.findall(r"[a-zA-Z\-]{4,}", text) if t not in {"candidate", "city", "official", "latest", "public"}]
    # preserve order, unique
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:8]


def ngram_max_repeat(words: List[str], n: int = 3) -> int:
    if len(words) < n:
        return 0
    grams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    return Counter(grams).most_common(1)[0][1]


def detect_tactic(text: str, verdict: str, source_cue: bool, blue_overlap: int) -> str:
    low = text.lower()
    has_quote = any(re.search(pat, text, re.I) for pat in QUOTE_PATTERNS)
    has_number = any(re.search(pat, text, re.I) for pat in NUMBER_PATTERNS)
    has_media = bool(re.search(r"\b(video|screenshot|clip|photo|larawan|infographic|graph|collage|cropped)\b", low))
    has_endorse = bool(re.search(r"\b(endorse|endorsement|suporta|backed by|allied with)\b", low))
    has_accuse = bool(re.search(r"\b(corrupt|nakaw|drug|crime|scandal|case|kaso|fraternity|suspek|shooting)\b", low))
    if verdict == "real":
        return "traceable_update" if source_cue else "ambiguous_or_uncertain"
    if verdict == "neutral":
        return "ambiguous_or_uncertain"
    if has_endorse:
        return "false_endorsement"
    if has_media:
        return "contextless_media"
    if has_quote and not source_cue:
        return "unverified_quote"
    if has_number and not source_cue:
        return "unsupported_number"
    if has_accuse:
        return "smear_or_accusation"
    if blue_overlap > 0:
        return "policy_distortion"
    if "viral" in low or "trending" in low:
        return "virality_without_evidence"
    return "ambiguous_or_uncertain"


def verdict_of(card: Dict[str, Any]) -> str:
    verdict = str(card.get("verdict", "") or "").lower()
    if verdict in {"real", "fake", "neutral"}:
        return verdict
    p_fake = card.get("p_fake", card.get("fake_likelihood_percent", 0.0))
    try:
        p_fake = float(p_fake)
    except Exception:
        p_fake = 0.5
    if p_fake > 1.0:
        p_fake = p_fake / 100.0
    if p_fake < 0.50:
        return "real"
    if p_fake > 0.60:
        return "fake"
    return "neutral"


def confidence_bucket(card: Dict[str, Any]) -> str:
    vals = []
    det = card.get("detectors", {})
    if isinstance(det, dict):
        for k in ("p_roberta_fake", "p_distil_fake", "p_degnn_fake", "p_ensemble_fake"):
            try:
                vals.append(float(det.get(k)))
            except Exception:
                pass
    if not vals:
        try:
            vals = [float(card.get("p_fake", 0.5))]
        except Exception:
            vals = [0.5]
    center_dist = abs(sum(vals) / len(vals) - 0.5)
    if center_dist >= 0.35:
        return "high"
    if center_dist >= 0.20:
        return "medium"
    return "low"


def detector_agreement(card: Dict[str, Any]) -> str:
    det = card.get("detectors", {})
    vals = []
    if isinstance(det, dict):
        for k in ("p_roberta_fake", "p_distil_fake", "p_degnn_fake"):
            try:
                vals.append(float(det.get(k)))
            except Exception:
                pass
    if len(vals) < 2:
        return "unknown"
    labels = [v >= 0.5 for v in vals]
    return "strong" if all(l == labels[0] for l in labels) else "mixed"


def summarize_model_bridge(card: Dict[str, Any], verdict: str) -> str:
    agreement = detector_agreement(card)
    conf = confidence_bucket(card)
    if agreement == "strong" and conf == "high":
        return "The AI detectors were strongly aligned, so the system can coach a clearer verdict. Students should still focus on the visible evidence, not only the score."
    if agreement == "mixed":
        return "The AI detectors did not fully agree, so the system should coach caution and verification instead of overconfidence."
    return "The AI signals were usable but not enough on their own. The lesson should come from the evidence and verification steps shown to the student."


def quality_features(card: Dict[str, Any], profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    text = str(card.get("text", "") or "")
    low = text.lower()
    words = re.findall(r"\w+", low)
    target = normalize_target(card)
    profile = profiles.get(target, {})
    election_hits = [term for term in GENERAL_ELECTION_TERMS if term in low]
    platform_keywords = [str(x).lower() for x in profile.get("platform_keywords", [])]
    topic_hits = [kw for kw in platform_keywords if kw and kw in low]
    blue_keys = extract_blue_truth_keywords(card)
    blue_overlap = sum(1 for kw in blue_keys if kw in low)
    source_hits = [pat for pat in SOURCE_PATTERNS if re.search(pat, low)]
    emotion_hits = [pat for pat in EMOTION_PATTERNS if re.search(pat, low)]
    offtopic_hits = [name for name, pat in OFFTOPIC_PATTERNS.items() if re.search(pat, text, re.I)]

    linked_blue_truth_present = isinstance(card.get("linked_blue_truth"), dict) and bool(card.get("linked_blue_truth", {}).get("id"))
    classification = card.get("classification", {})
    has_explicit_target = bool(card.get("targets")) or (isinstance(classification, dict) and bool(classification.get("targets"))) or bool(card.get("linked_blue_truth", {}).get("candidate"))
    generation_mode = str(card.get("metadata", {}).get("generation_mode", "") or "")

    feat = {
        "target": target,
        "word_count": len(words),
        "election_hit_count": len(set(election_hits)),
        "election_hits": sorted(set(election_hits)),
        "topic_hits": topic_hits,
        "topic_hit_count": len(topic_hits),
        "blue_truth_overlap": blue_overlap,
        "linked_blue_truth_present": linked_blue_truth_present,
        "has_explicit_target": has_explicit_target,
        "generation_mode": generation_mode,
        "source_hit_count": len(source_hits),
        "emotion_hit_count": len(emotion_hits),
        "offtopic_hits": offtopic_hits,
        "quote_unbalanced": text.count('"') % 2 == 1,
        "candidate_placeholder_count": len(re.findall(r"\bCandidate\s+[A-Z]{1,5}\b", text)),
        "entity_placeholder_count": len(re.findall(r"\bEntity\b", text)),
        "ngram_max_repeat": ngram_max_repeat(words, 3),
        "lexical_diversity": round((len(set(words)) / max(1, len(words))), 4),
        "truncated_or_malformed": bool(MALFORMED_ENDING_RE.search(text.strip())),
    }

    score = 0.0
    score += min(0.25, feat["election_hit_count"] * 0.05)
    score += min(0.25, feat["topic_hit_count"] * 0.08)
    score += min(0.10, feat["blue_truth_overlap"] * 0.04)
    score += 0.10 if feat["source_hit_count"] > 0 else 0.0
    score += 0.05 if feat["word_count"] >= 60 else (0.02 if feat["word_count"] >= 40 else (0.01 if feat["word_count"] >= 24 else 0.0))
    score += 0.05 if feat["lexical_diversity"] >= 0.50 else 0.0
    score += 0.05 if feat["linked_blue_truth_present"] else 0.0
    score += 0.05 if feat["has_explicit_target"] else 0.0
    score += 0.08 if feat["generation_mode"] == "rule_constrained_template" else 0.0

    if feat["quote_unbalanced"]:
        score -= 0.18
    if feat["candidate_placeholder_count"] >= 3:
        score -= 0.15
    if feat["entity_placeholder_count"] >= 4:
        score -= 0.10
    if feat["ngram_max_repeat"] >= 3:
        score -= 0.18
    if feat["truncated_or_malformed"]:
        score -= 0.18
    if "sports" in feat["offtopic_hits"]:
        score -= 0.25
    if "showbiz" in feat["offtopic_hits"]:
        score -= 0.25
    if "transport_util" in feat["offtopic_hits"]:
        score -= 0.18
    if "violent_incident" in feat["offtopic_hits"] and feat["topic_hit_count"] == 0:
        score -= 0.20

    feat["quality_score"] = round(max(0.0, min(1.0, score)), 4)
    weak_theme = (
        feat["election_hit_count"] < 1 and
        feat["topic_hit_count"] == 0 and
        feat["blue_truth_overlap"] == 0 and
        not feat["linked_blue_truth_present"]
    )
    feat["critical_fail"] = (
        weak_theme or
        (feat["topic_hit_count"] == 0 and feat["blue_truth_overlap"] == 0 and not feat["linked_blue_truth_present"] and not feat["has_explicit_target"]) or
        feat["truncated_or_malformed"] or
        ("sports" in feat["offtopic_hits"]) or
        ("showbiz" in feat["offtopic_hits"])
    )
    return feat


def rejection_reasons(features: Dict[str, Any]) -> List[str]:
    reasons = []
    if features["election_hit_count"] < 2:
        reasons.append("weak election context")
    if features["topic_hit_count"] == 0 and features["blue_truth_overlap"] == 0:
        reasons.append("not aligned with the target candidate's known issues or linked blue truth")
    if features["quote_unbalanced"]:
        reasons.append("unbalanced quote suggests truncation or malformed text")
    if features["candidate_placeholder_count"] >= 3:
        reasons.append("too many candidate placeholders make the post hard to follow")
    if features["entity_placeholder_count"] >= 4:
        reasons.append("too many redacted entities reduce clarity")
    if features["ngram_max_repeat"] >= 3:
        reasons.append("repeated wording suggests low-quality generation")
    if features["truncated_or_malformed"]:
        reasons.append("text appears truncated or unfinished")
    if "sports" in features["offtopic_hits"]:
        reasons.append("sports content is off-theme for the election simulator")
    if "showbiz" in features["offtopic_hits"]:
        reasons.append("showbiz/celebrity content is off-theme for the election simulator")
    if "transport_util" in features["offtopic_hits"]:
        reasons.append("utility or transport content appears unrelated to the candidate storyline")
    if "violent_incident" in features["offtopic_hits"] and features["topic_hit_count"] == 0:
        reasons.append("violent-incident claim lacks clear campaign or policy relevance")
    if features["quality_score"] < 0.45:
        reasons.append("overall quality score is below the release threshold")
    return reasons


def extract_evidence_snippet(text: str, pattern: str, fallback_len: int = 120) -> str:
    m = re.search(pattern, text, re.I)
    if m:
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 60)
        return text[start:end].strip()
    return text[:fallback_len].strip()


def build_red_flags(card: Dict[str, Any], features: Dict[str, Any], tactic: str, profiles: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    text = str(card.get("text", "") or "")
    flags: List[Dict[str, str]] = []
    if features["emotion_hit_count"] > 0:
        flags.append({
            "flag": "Emotional or urgent framing",
            "evidence": extract_evidence_snippet(text, EMOTION_PATTERNS[0]),
            "why_it_matters": "Urgent wording pushes fast reactions before careful checking.",
            "tip": "Pause before trusting or sharing. Verify first.",
        })
    if tactic == "unverified_quote" or features["quote_unbalanced"]:
        flags.append({
            "flag": "Quote without reliable context",
            "evidence": extract_evidence_snippet(text, QUOTE_PATTERNS[0]),
            "why_it_matters": "A quote needs a full transcript, speech, press release, or credible report.",
            "tip": "Search the exact quote with the candidate name and an official source.",
        })
    if tactic == "unsupported_number":
        flags.append({
            "flag": "Number used without a visible source",
            "evidence": extract_evidence_snippet(text, NUMBER_PATTERNS[0]),
            "why_it_matters": "Numbers can mislead when the method, date, or source is missing.",
            "tip": "Find the report or document that produced the number.",
        })
    if tactic == "false_endorsement":
        flags.append({
            "flag": "Claimed endorsement or alliance",
            "evidence": extract_evidence_snippet(text, r"\b(endorse|endorsement|suporta|backed by|allied with)\b"),
            "why_it_matters": "Political support should be traceable through official statements or filings.",
            "tip": "Check the endorser's own verified page or public filing.",
        })
    if tactic == "contextless_media":
        flags.append({
            "flag": "Media clip without full context",
            "evidence": extract_evidence_snippet(text, r"\b(video|screenshot|clip|photo|larawan|infographic|graph|collage|cropped)\b"),
            "why_it_matters": "A short clip or repost can hide when, where, and why it happened.",
            "tip": "Trace the media back to the original upload and full context.",
        })
    if tactic == "smear_or_accusation":
        flags.append({
            "flag": "Serious accusation with unclear evidence",
            "evidence": extract_evidence_snippet(text, r"\b(corrupt|nakaw|drug|crime|scandal|case|kaso|fraternity|suspek|shooting)\b"),
            "why_it_matters": "Accusations need documentary or official confirmation, not only repetition.",
            "tip": "Look for a police report, court document, or verified statement.",
        })
    if features["candidate_placeholder_count"] >= 3:
        flags.append({
            "flag": "Too many unnamed or placeholder actors",
            "evidence": "Multiple placeholder candidate names appear in one short post.",
            "why_it_matters": "When the actors are unclear, students cannot verify who actually said or did what.",
            "tip": "Ask which person, office, or institution is being referenced and trace each one.",
        })
    if not flags:
        flags.append({
            "flag": "Needs corroboration",
            "evidence": text[:110].strip(),
            "why_it_matters": "Even a plausible-looking post should be checked against a traceable source.",
            "tip": "Open a reliable source before deciding to trust or share.",
        })
    return flags[:3]


def build_positive_signals(card: Dict[str, Any], features: Dict[str, Any], verdict: str) -> List[Dict[str, str]]:
    text = str(card.get("text", "") or "")
    signals: List[Dict[str, str]] = []
    if features["source_hit_count"] > 0:
        signals.append({
            "signal": "Mentions a traceable source",
            "evidence": extract_evidence_snippet(text, SOURCE_PATTERNS[0]),
            "why_it_matters": "Source mentions create a path for verification.",
        })
    if features["blue_truth_overlap"] > 0:
        signals.append({
            "signal": "Matches the linked campaign context",
            "evidence": "The post overlaps with the card's linked blue truth or known candidate issue.",
            "why_it_matters": "Scenario coherence helps the player learn from meaningful cases rather than random text.",
        })
    if verdict == "real" and not signals:
        signals.append({
            "signal": "No strong manipulation cue detected",
            "evidence": "The text does not rely heavily on urgency, smear framing, or unsupported viral language.",
            "why_it_matters": "Absence of obvious red flags is not proof, but it lowers immediate suspicion.",
        })
    return signals[:2]


def build_analysis_steps(card: Dict[str, Any], tactic_info: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, str]]:
    trusted_sources = profile.get("trusted_sources", [])[:2]
    source_hint = ", or ".join(trusted_sources) if trusted_sources else "official and independent sources"
    steps = tactic_info.get("default_steps", [])
    return [
        {
            "question": "Who is behind this information?",
            "action": f"Identify the page, uploader, or speaker. Ask whether it is official, campaign-affiliated, anonymous, or only a repost. Prefer {source_hint}.",
        },
        {
            "question": "What is the evidence?",
            "action": steps[0] if steps else "Find the document, clip, report, or statement that directly supports the claim.",
        },
        {
            "question": "What do other sources say?",
            "action": steps[1] if len(steps) > 1 else "Cross-check with at least two credible sources or the official bulletin/website.",
        },
        {
            "question": "What is the original context?",
            "action": steps[2] if len(steps) > 2 else "Trace the quote, number, or media back to its original context before you decide.",
        },
    ]


def build_feedback_strings(seed: str, verdict: str, tactic: str, profile: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, str]:
    name = profile.get("name", f"Candidate {features['target']}")
    role = profile.get("role", "candidate")
    focus = {
        "unverified_quote": "verify the original quote",
        "unsupported_number": "check the source of the number",
        "contextless_media": "recover the missing context",
        "false_endorsement": "confirm the endorsement record",
        "smear_or_accusation": "separate allegation from evidence",
        "policy_distortion": "compare the post with the actual policy record",
        "virality_without_evidence": "ignore virality and inspect evidence",
        "traceable_update": "confirm the cited source before trusting the update",
        "ambiguous_or_uncertain": "slow down and verify the unclear part",
    }[tactic]

    feed_up_pool = [
        f"Your goal in this card is to {focus} before deciding whether to trust, flag, ignore, or share.",
        f"Start with the verification task for this card: {focus}. Only decide after you inspect the evidence path.",
        f"In this scenario, your first job is to {focus}. The score matters less than the verification routine you use.",
    ]

    if verdict == "fake":
        feedback_pool = [
            f"This post about {name} ({role}) shows cues that match misinformation patterns: the problem is not only what it claims, but how weakly the claim is supported.",
            f"The claim about {name} ({role}) is not release-safe to trust because the evidence trail is thin, missing, or distorted.",
            f"For this {name} ({role}) post, the warning signs come from weak sourcing and misleading framing, not only from the headline claim itself.",
        ]
        feed_forward_pool = [
            "Do not rely on the post alone. Open a source, compare the wording, and verify the evidence before taking action.",
            "Slow the decision down: trace the source, inspect the evidence, and look for independent confirmation before you react.",
            "Treat this like a verification drill. Separate the allegation from the evidence, then confirm the original source before you decide.",
        ]
    elif verdict == "real":
        feedback_pool = [
            f"This post about {name} ({role}) is more plausible because it has better traceability than a typical rumor, but students should still verify the source path.",
            f"The update about {name} ({role}) looks stronger than a rumor because it points toward a checkable record, source, or official notice.",
            f"This {name} ({role}) post is closer to a credible update than a rumor, yet the right lesson is still verification, not blind trust.",
        ]
        feed_forward_pool = [
            "Treat the post as checkable, not automatically trustworthy. Confirm the source, then compare with another reliable outlet.",
            "Use the post as a starting point for verification: open the cited source, then see whether a second reliable source matches it.",
            "Even when a card looks credible, verify the document path and compare it with another trustworthy source before acting on it.",
        ]
    else:
        feedback_pool = [
            f"This post about {name} ({role}) is too uncertain for a confident judgment. The safest move is careful verification before reacting.",
            f"The post about {name} ({role}) mixes a plausible topic with incomplete evidence, so uncertainty is the correct first response.",
            f"For this {name} ({role}) card, the right lesson is not forced certainty. The evidence is still incomplete, so verification comes first.",
        ]
        feed_forward_pool = [
            "Use a verify-first response: delay sharing, find the missing source, and compare with official or independent coverage.",
            "When the evidence is incomplete, pause the judgment. Find the missing document, quote, or source before taking a side.",
            "Respond with caution: do not overcommit to trust or flag until you have checked the unclear part against a reliable source.",
        ]
    return {
        "feed_up": choose(seed + ':feed_up', feed_up_pool),
        "feedback": choose(seed + ':feedback', feedback_pool),
        "feed_forward": choose(seed + ':feed_forward', feed_forward_pool),
    }


def build_student_prompt(card_id: str, tactic: str) -> str:
    pools = {
        "unverified_quote": [
            "What source would settle this quote fastest: a transcript, a full video, or a repost?",
            "Which exact words would you search to find the original quote?",
            "If this quote were edited, what clue in the post would make that harder to notice?",
        ],
        "unsupported_number": [
            "What document or report should exist if this number is real?",
            "What is the first number in the post that needs a source before you trust it?",
            "How could a real number still be misleading if its timeframe is hidden?",
        ],
        "contextless_media": [
            "What is missing from this clip or screenshot: date, location, or full sequence?",
            "If this media were cropped, what important context could be hiding outside the frame?",
            "Which original upload would you want to find before judging the post?",
        ],
        "false_endorsement": [
            "Whose official statement would confirm or deny this alliance?",
            "What public record would make this endorsement claim checkable?",
            "Why is a repost weaker evidence than a verified endorsement announcement?",
        ],
        "smear_or_accusation": [
            "What evidence would turn this accusation from rumor into a checkable claim?",
            "Which part is fact and which part is interpretation or insinuation?",
            "What official document would you look for before believing this accusation?",
        ],
        "policy_distortion": [
            "What changed here: timing, scope, approval status, or cost?",
            "Which line in the real proposal would you compare against the post?",
            "How can a true policy topic still be presented in a misleading way?",
        ],
        "virality_without_evidence": [
            "Why do likes and comments not count as proof?",
            "What is the earliest source you need to find before taking this post seriously?",
            "How can a viral post still be wrong?",
        ],
        "traceable_update": [
            "Which source in the post should you open first, and why?",
            "What second source would you use to confirm this update?",
            "What makes this post more checkable than a rumor-only post?",
        ],
        "ambiguous_or_uncertain": [
            "What part of the claim is still missing evidence?",
            "What single source would reduce the uncertainty most?",
            "When the evidence is incomplete, what is the safest action inside the game?",
        ],
    }
    return choose(card_id + tactic, pools[tactic])


def build_reflection_prompt(card_id: str, verdict: str) -> str:
    pools = {
        "fake": [
            "Which clue pushed you toward flagging this post: source weakness, missing context, or manipulative framing?",
            "What almost made this post believable, and how did you correct for it?",
            "Which verification move would protect you from a similar post in a real feed?",
        ],
        "real": [
            "What made this post more trustworthy than a typical rumor?",
            "Which source path helped you separate a plausible update from a fake one?",
            "How would you explain to a classmate why this post still needed verification even if it turned out credible?",
        ],
        "neutral": [
            "What kept you from making a confident judgment right away?",
            "Which missing detail would have changed your decision the most?",
            "Why is uncertainty a valid outcome when evidence is incomplete?",
        ],
    }
    return choose(card_id + verdict, pools[verdict])


def build_teaching(card: Dict[str, Any], features: Dict[str, Any], bank: Dict[str, Any], profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    verdict = verdict_of(card)
    tactic = detect_tactic(str(card.get("text", "")), verdict, features["source_hit_count"] > 0, features["blue_truth_overlap"])
    tactic_info = bank["tactics"][tactic]
    profile = profiles.get(features["target"], {})
    feedback = build_feedback_strings(str(card.get("id", "")), verdict, tactic, profile, features)

    ai_conf = confidence_bucket(card)
    agreement = detector_agreement(card)

    return {
        "lesson_focus": {
            "code": tactic,
            "title": tactic_info["title"],
            "why_it_matters": tactic_info["why_it_matters"],
        },
        "feed_up": feedback["feed_up"],
        "feedback": feedback["feedback"],
        "feed_forward": feedback["feed_forward"],
        "red_flags": build_red_flags(card, features, tactic, profiles),
        "positive_signals": build_positive_signals(card, features, verdict),
        "analysis_steps": build_analysis_steps(card, tactic_info, profile),
        "student_prompt": build_student_prompt(str(card.get("id", "")), tactic),
        "reflection_prompt": build_reflection_prompt(str(card.get("id", "")), verdict),
        "real_time_transfer": "Use the same routine outside the game: pause, trace the source, inspect the evidence, and cross-check before you react or share.",
        "frameworks": {
            "civic_online_reasoning": bank["frameworks"]["civic_online_reasoning"],
            "sift": bank["frameworks"]["sift"],
        },
        "ai_bridge": {
            "detector_agreement": agreement,
            "confidence_bucket": ai_conf,
            "student_safe_explanation": summarize_model_bridge(card, verdict),
            "do_not_overtrust_ai": "The score is a guide for learning, not a replacement for verification.",
        },
    }


def curate(cards: List[Dict[str, Any]], profiles: Dict[str, Dict[str, Any]], bank: Dict[str, Any], min_quality_score: float = 0.45) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    issue_counter: Counter[str] = Counter()
    tactic_counter: Counter[str] = Counter()
    focus_counter: Counter[str] = Counter()
    verdict_counter: Counter[str] = Counter()
    per_candidate = Counter()

    for card in cards:
        features = quality_features(card, profiles)
        reasons = rejection_reasons(features)
        if features["critical_fail"] or features["quality_score"] < min_quality_score:
            issue_counter.update(reasons)
            row = dict(card)
            row["quality"] = features
            row["rejection_reasons"] = reasons
            rejected.append(row)
            continue

        row = dict(card)
        teaching = build_teaching(card, features, bank, profiles)
        verdict = verdict_of(card)
        tactic = teaching["lesson_focus"]["code"]
        row["quality"] = features
        row["teaching"] = teaching
        row["classification"] = dict(row.get("classification", {}))
        row["classification"]["explanation"] = teaching["feedback"]
        row["classification"]["teaching_focus"] = tactic
        kept.append(row)

        tactic_counter[tactic] += 1
        focus_counter[teaching["lesson_focus"]["title"]] += 1
        verdict_counter[verdict] += 1
        per_candidate[features["target"]] += 1

    report = {
        "input_cards": len(cards),
        "kept_cards": len(kept),
        "rejected_cards": len(rejected),
        "keep_rate": round(len(kept) / max(1, len(cards)), 4),
        "rejection_reasons": dict(issue_counter.most_common()),
        "lesson_focus_counts": dict(focus_counter.most_common()),
        "tactic_counts": dict(tactic_counter.most_common()),
        "verdict_counts": dict(verdict_counter.most_common()),
        "candidate_counts": dict(per_candidate.most_common()),
    }
    return kept, rejected, report


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Curate story cards for classroom use: reject weak cards and add diverse student-facing teaching feedback."
    )
    ap.add_argument("--in_file", required=True, help="Input story_cards or themed unity cards JSON.")
    ap.add_argument("--out_file", required=True, help="Curated output JSON.")
    ap.add_argument("--reject_out", default=None, help="Optional rejected-cards JSON.")
    ap.add_argument("--report_out", default=None, help="Optional report JSON.")
    ap.add_argument("--candidate_profiles", default=None, help="Optional rich candidate profiles JSON.")
    ap.add_argument("--teaching_bank", default=None, help="Optional teaching-response bank JSON.")
    ap.add_argument("--min_quality_score", type=float, default=0.45)
    args = ap.parse_args()

    cards = read_json(Path(args.in_file))
    if not isinstance(cards, list):
        raise ValueError("Input JSON must be a list of card objects.")

    profiles = load_profiles(Path(args.candidate_profiles) if args.candidate_profiles else None)
    bank = load_bank(Path(args.teaching_bank) if args.teaching_bank else None)

    kept, rejected, report = curate(cards, profiles, bank, min_quality_score=float(args.min_quality_score))

    write_json(Path(args.out_file), kept)
    if args.reject_out:
        write_json(Path(args.reject_out), rejected)
    if args.report_out:
        write_json(Path(args.report_out), report)

    print(f"[OK] Curated cards written to {args.out_file} (kept={len(kept)}, rejected={len(rejected)})")


if __name__ == "__main__":
    main()

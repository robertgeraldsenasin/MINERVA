from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported JSON structure in {path}; expected a list of cards.")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_target_list(card: Dict[str, Any], candidate_profiles: Dict[str, Any]) -> List[str]:
    for key in ("targets", "target", "candidate", "candidate_id"):
        if key in card and card[key]:
            val = card[key]
            if isinstance(val, list):
                return [str(x) for x in val]
            return [str(val)]
    # fallback keyword matching on text
    text = str(card.get("text", ""))
    matches = []
    for cand_id, profile in candidate_profiles.items():
        keywords = [cand_id, profile.get("name", "")]
        for kw in keywords:
            kw = str(kw).strip()
            if kw and kw.lower() in text.lower():
                matches.append(cand_id)
                break
    return matches or ["A"]


def choose_blue_truth(blue_truths: List[Dict[str, Any]], targets: List[str], rng: random.Random) -> Dict[str, Any]:
    preferred = [bt for bt in blue_truths if not bt.get("candidate") or str(bt.get("candidate")) in targets]
    pool = preferred or blue_truths
    return rng.choice(pool)


def build_classification(card: Dict[str, Any], blue_truth: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
    verdict = str(card.get("verdict", "")).strip().lower()
    if not verdict:
        p_fake = float(card.get("p_fake", card.get("fake_likelihood_percent", 0.0)) or 0.0)
        if p_fake > 1.0:
            p_fake = p_fake / 100.0
        if p_fake < 0.50:
            verdict = "real"
        elif p_fake > 0.60:
            verdict = "fake"
        else:
            verdict = "neutral"

    is_misinfo = verdict == "fake"
    truth_type = "red" if verdict in {"fake", "neutral"} else "blue"

    explanation_obj = card.get("explanation", {})
    if isinstance(explanation_obj, dict):
        explanation = explanation_obj.get("summary") or "Generated explanation not available."
    else:
        explanation = str(explanation_obj) if explanation_obj else "Generated explanation not available."

    return {
        "truth_type": truth_type,
        "is_misinformation": bool(is_misinfo),
        "targets": targets,
        "linked_blue_truth_id": blue_truth.get("id"),
        "explanation": explanation,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert generic Unity cards into the current MINERVA story-card schema."
    )
    ap.add_argument("--in_file", required=True, help="Input Unity cards (.json or .jsonl).")
    ap.add_argument("--blue_truths", required=True, help="Blue truths template JSON.")
    ap.add_argument("--candidate_profiles", required=True, help="Candidate profiles template JSON.")
    ap.add_argument("--out_file", default="generated/story_cards.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--days", type=int, default=14)
    args = ap.parse_args()

    in_path = Path(args.in_file)
    blue_path = Path(args.blue_truths)
    cand_path = Path(args.candidate_profiles)

    if not in_path.exists():
        raise FileNotFoundError(in_path)
    if not blue_path.exists():
        raise FileNotFoundError(blue_path)
    if not cand_path.exists():
        raise FileNotFoundError(cand_path)

    cards = read_json_or_jsonl(in_path)
    blue_truths = json.loads(blue_path.read_text(encoding="utf-8"))
    candidate_profiles = json.loads(cand_path.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    out_rows = []
    for idx, card in enumerate(cards, start=1):
        targets = normalize_target_list(card, candidate_profiles)
        blue_truth = choose_blue_truth(blue_truths, targets, rng)
        day = ((idx - 1) % max(1, int(args.days))) + 1

        row = dict(card)
        row["day"] = row.get("day", day)
        row["targets"] = targets
        row["linked_blue_truth"] = blue_truth
        row["classification"] = build_classification(card, blue_truth, targets)

        out_rows.append(row)

    write_json(Path(args.out_file), out_rows)
    print(f"[OK] Wrote story cards -> {args.out_file} (rows={len(out_rows)})")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
MINERVA Script 21: Card balancer for Unity

Goal
----
Take a *scored* JSONL (or JSON) file and export an EXACTLY balanced set of cards:
REAL / FAKE / NEUTRAL (e.g., 200/200/200), without retraining anything.

Typical inputs
--------------
- generated/gpt2_synthetic_scored.jsonl   (from script 13)
- generated/gpt2_synthetic_final.jsonl    (from script 13)
- generated/gpt2_synthetic_verdicts.json  (from script 18)

Classification rule (defaults)
------------------------------
We treat scores as P(fake):
  REAL    if p_fake <  neutral_low
  NEUTRAL if neutral_low <= p_fake <= neutral_high
  FAKE    if p_fake >  neutral_high

Defaults are neutral_low=0.50, neutral_high=0.60 (your requested 50–60% neutral band).
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # Common patterns: {"cards":[...]} or {"data":[...]}
        for k in ("cards", "data", "items", "rows"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
    raise ValueError(
        f"Unsupported JSON structure in {path} (expected list or dict with a list field).")


def read_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    return _read_json(path)


def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValueError(
            f"Expected a number, got {type(x).__name__}: {x!r}") from e


def get_p_fake(rec: Dict[str, Any], score_key: str) -> float:
    """
    Best-effort extraction of 'probability of fake' from a record.

    Priority:
      1) rec[score_key] (default: p_qlattice_fake)
      2) rec['p_qlattice_fake']
      3) rec['p_fake']
      4) rec['fake_likelihood_percent'] / 100
      5) rec['qlattice']['score'] if it looks like a probability
      6) average of detector probs if present (p_roberta_fake + p_distil_fake)/2
    """
    for k in (score_key, "p_qlattice_fake", "p_fake"):
        if k in rec and rec[k] is not None:
            p = _as_float(rec[k])
            return max(0.0, min(1.0, p))

    if "fake_likelihood_percent" in rec and rec["fake_likelihood_percent"] is not None:
        p = _as_float(rec["fake_likelihood_percent"]) / 100.0
        return max(0.0, min(1.0, p))

    q = rec.get("qlattice")
    if isinstance(q, dict) and q.get("score") is not None:
        s = _as_float(q["score"])
        if 0.0 <= s <= 1.0:
            return s

    pr = rec.get("p_roberta_fake")
    pd = rec.get("p_distil_fake", rec.get("p_distilbert_fake"))
    if pr is not None and pd is not None:
        p = 0.5 * (_as_float(pr) + _as_float(pd))
        return max(0.0, min(1.0, p))

    raise KeyError(
        f"Could not find a usable fake-probability in record. "
        f"Tried keys: {score_key}, p_qlattice_fake, p_fake, fake_likelihood_percent, qlattice.score, detector probs."
    )


def classify(p_fake: float, neutral_low: float, neutral_high: float) -> str:
    if p_fake < neutral_low:
        return "real"
    if p_fake > neutral_high:
        return "fake"
    return "neutral"


def _ensure_minimal_fields(rec: Dict[str, Any], idx: int, p_fake: float, verdict: str) -> Dict[str, Any]:
    out = dict(rec)  # keep everything; Unity can ignore extra keys
    if "id" not in out or out["id"] is None or str(out["id"]).strip() == "":
        out["id"] = f"card_{idx:06d}"

    # Normalize text presence
    if "text" not in out:
        # Try common fallback keys
        for k in ("content", "post", "article", "body"):
            if k in out and out[k] is not None:
                out["text"] = str(out[k])
                break
        out.setdefault("text", "")

    out["verdict"] = verdict
    out["p_fake"] = float(p_fake)

    # Percent fields are convenient for UI (and match Script 18 names)
    out.setdefault("fake_likelihood_percent", float(100.0 * p_fake))
    out.setdefault("credibility_percent", float(100.0 * (1.0 - p_fake)))

    # If Script 18 explanation exists but we changed verdict to neutral, adjust summary lightly.
    if verdict == "neutral" and isinstance(out.get("explanation"), dict):
        expl = dict(out["explanation"])
        expl["summary"] = (
            f"Verdict: NEUTRAL (estimated fake-likelihood {100.0 * p_fake:.1f}%). "
            "The score is near the decision boundary, so treat it as uncertain and verify carefully."
        )
        # Keep other keys if present
        out["explanation"] = expl

    return out


def _sample_group(
    items: List[Dict[str, Any]],
    n: int,
    rng: random.Random,
    allow_reuse: bool,
    strategy: str,
    sort_key,
) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if not items:
        raise ValueError("Cannot sample from an empty class bucket.")

    if strategy == "ranked":
        ranked = sorted(items, key=sort_key)
        if len(ranked) >= n:
            return ranked[:n]
        if not allow_reuse:
            raise ValueError(
                f"Not enough unique items for class (need {n}, have {len(ranked)}).")
        # Reuse by cycling (deterministic, avoids RNG bias)
        out: List[Dict[str, Any]] = []
        i = 0
        while len(out) < n:
            out.append(ranked[i % len(ranked)])
            i += 1
        return out

    # strategy == random
    if len(items) >= n:
        return rng.sample(items, n)
    if not allow_reuse:
        raise ValueError(
            f"Not enough unique items for class (need {n}, have {len(items)}).")
    return [rng.choice(items) for _ in range(n)]


def write_output(path: Path, rows: List[Dict[str, Any]], out_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if out_format == "jsonl" or path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(rows, ensure_ascii=False,
                        indent=2), encoding="utf-8")


def main() -> None:
    repo_root = _repo_root()
    os.chdir(repo_root)

    ap = argparse.ArgumentParser(
        description="Balance scored MINERVA cards for Unity (REAL/FAKE/NEUTRAL).")
    ap.add_argument("--in_file", default="generated/gpt2_synthetic_scored.jsonl",
                    help="Input scored file (.jsonl or .json).")
    ap.add_argument("--out_file", default="generated/unity_cards_balanced.json",
                    help="Output file for Unity (.json or .jsonl).")
    ap.add_argument("--out_format", choices=["json", "jsonl"], default="json",
                    help="Output format (json array is Unity-friendly).")

    ap.add_argument("--n_per_class", type=int, default=200,
                    help="How many cards per class (REAL/FAKE/NEUTRAL).")
    ap.add_argument("--n_real", type=int, default=None)
    ap.add_argument("--n_fake", type=int, default=None)
    ap.add_argument("--n_neutral", type=int, default=None)

    ap.add_argument("--neutral_low", type=float, default=0.50,
                    help="Lower bound of NEUTRAL band (inclusive). REAL is below this.")
    ap.add_argument("--neutral_high", type=float, default=0.60,
                    help="Upper bound of NEUTRAL band (inclusive). FAKE is above this.")

    ap.add_argument("--score_key", default="p_qlattice_fake",
                    help="Preferred score key to read from each record (probability of fake).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategy", choices=["ranked", "random"], default="ranked",
                    help="Selection strategy per class. ranked=deterministic (extremes/closest), random=random sample.")
    ap.add_argument("--allow_reuse", action="store_true",
                    help="If a class bucket is too small, allow reusing items (duplicates) to hit exact counts.")
    ap.add_argument("--shuffle", action="store_true",
                    help="Shuffle final output (useful for gameplay variety).")

    args = ap.parse_args()

    neutral_low = float(args.neutral_low)
    neutral_high = float(args.neutral_high)
    if not (0.0 <= neutral_low <= neutral_high <= 1.0):
        raise ValueError(
            "--neutral_low and --neutral_high must satisfy 0 <= low <= high <= 1.")

    n_real = int(args.n_real) if args.n_real is not None else int(
        args.n_per_class)
    n_fake = int(args.n_fake) if args.n_fake is not None else int(
        args.n_per_class)
    n_neu = int(args.n_neutral) if args.n_neutral is not None else int(
        args.n_per_class)

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)

    rows_in = read_records(in_path)
    if not rows_in:
        raise RuntimeError(f"Input file is empty: {in_path}")

    # Classify + normalize fields
    bucket: Dict[str, List[Dict[str, Any]]] = {
        "real": [], "fake": [], "neutral": []}

    for i, rec in enumerate(rows_in):
        p_fake = get_p_fake(rec, score_key=args.score_key)
        verdict = classify(p_fake, neutral_low=neutral_low,
                           neutral_high=neutral_high)
        norm = _ensure_minimal_fields(
            rec, idx=i, p_fake=p_fake, verdict=verdict)
        bucket[verdict].append(norm)

    print(f"[21] Loaded {len(rows_in)} rows from {in_path}")
    print("[21] Available by class:",
          {k: len(v) for k, v in bucket.items()},
          f"(neutral band: {neutral_low:.2f}–{neutral_high:.2f})")

    rng = random.Random(int(args.seed))

    # ranked sort keys:
    #  - real: smallest p_fake first (most real)
    #  - fake: largest p_fake first (most fake) => use negative p_fake
    #  - neutral: closest to 0.5 first (most ambiguous)
    pick_real = _sample_group(
        bucket["real"],
        n=n_real,
        rng=rng,
        allow_reuse=bool(args.allow_reuse),
        strategy=args.strategy,
        sort_key=lambda r: r.get("p_fake", 0.0),
    )
    pick_fake = _sample_group(
        bucket["fake"],
        n=n_fake,
        rng=rng,
        allow_reuse=bool(args.allow_reuse),
        strategy=args.strategy,
        sort_key=lambda r: -float(r.get("p_fake", 0.0)),
    )
    pick_neu = _sample_group(
        bucket["neutral"],
        n=n_neu,
        rng=rng,
        allow_reuse=bool(args.allow_reuse),
        strategy=args.strategy,
        sort_key=lambda r: abs(float(r.get("p_fake", 0.0)) - 0.5),
    )

    out_rows = pick_real + pick_fake + pick_neu
    if args.shuffle:
        rng.shuffle(out_rows)

    write_output(out_path, out_rows, out_format=args.out_format)

    print("[21] Wrote balanced cards ->", out_path.resolve())
    print("[21] Output counts:",
          {"real": n_real, "fake": n_fake, "neutral": n_neu},
          "(total:", len(out_rows), ")")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[FATAL]", repr(e))
        sys.exit(1)

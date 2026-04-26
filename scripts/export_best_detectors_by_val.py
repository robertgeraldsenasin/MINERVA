from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def score_run(run: Dict[str, Any]) -> Tuple[float, float, float, float]:
    val = run.get("val", {}) or {}
    test = run.get("test", {}) or {}
    return (
        float(val.get("f1", -1.0)),
        float(val.get("accuracy", -1.0)),
        float(test.get("f1", -1.0)),
        float(test.get("accuracy", -1.0)),
    )


def pick_best(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        raise ValueError("No runs to choose from.")
    return sorted(runs, key=score_run, reverse=True)[0]


def copytree_overwrite(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source model dir: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export the best 5-seed detectors using VALIDATION metrics instead of TEST metrics."
    )
    ap.add_argument("--run_id", required=True, help="Run id used in Script 17")
    ap.add_argument("--reports_dir", default="reports")
    ap.add_argument("--models_dir", default="models")
    args = ap.parse_args()

    summary_path = Path(args.reports_dir) / f"detectors_5seed_summary_{args.run_id}.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary file: {summary_path}\nRun scripts/17_run_5seeds_detectors.py first."
        )

    summary = read_json(summary_path)
    roberta_runs = summary.get("roberta", {}).get("per_seed", [])
    distil_runs = summary.get("distilbert", {}).get("per_seed", [])

    best_roberta = pick_best(roberta_runs)
    best_distil = pick_best(distil_runs)

    legacy_roberta = Path(args.models_dir) / "roberta_finetuned"
    legacy_distil = Path(args.models_dir) / "distilbert_multilingual_finetuned"

    copytree_overwrite(Path(best_roberta["output_dir"]), legacy_roberta)
    copytree_overwrite(Path(best_distil["output_dir"]), legacy_distil)

    payload = {
        "run_id": args.run_id,
        "selection_policy": "validation_f1_then_validation_accuracy_then_test_f1_then_test_accuracy",
        "best_roberta": {
            "seed": best_roberta.get("seed"),
            "src": best_roberta.get("output_dir"),
            "dst": str(legacy_roberta),
            "val": best_roberta.get("val"),
            "test": best_roberta.get("test"),
        },
        "best_distilbert": {
            "seed": best_distil.get("seed"),
            "src": best_distil.get("output_dir"),
            "dst": str(legacy_distil),
            "val": best_distil.get("val"),
            "test": best_distil.get("test"),
        },
    }
    out_path = Path(args.reports_dir) / f"best_seed_export_by_val_{args.run_id}.json"
    write_json(out_path, payload)

    print("[OK] Exported validation-best detector checkpoints:")
    print(" -", legacy_roberta)
    print(" -", legacy_distil)
    print("[OK] Wrote:", out_path)


if __name__ == "__main__":
    main()

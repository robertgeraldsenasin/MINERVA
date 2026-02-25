from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import shutil
from typing import Any, Dict, List


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize(model_name: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # runs = list of metrics.json dicts
    test_f1 = [r["test"]["f1"] for r in runs]
    test_acc = [r["test"]["accuracy"] for r in runs]
    return {
        "model": model_name,
        "n_runs": len(runs),
        "test_f1_mean": statistics.mean(test_f1),
        "test_f1_std": statistics.pstdev(test_f1) if len(test_f1) > 1 else 0.0,
        "test_acc_mean": statistics.mean(test_acc),
        "test_acc_std": statistics.pstdev(test_acc) if len(test_acc) > 1 else 0.0,
        "per_seed": [
            {
                "seed": r.get("seed"),
                "output_dir": r.get("output_dir"),
                "test": r.get("test"),
            }
            for r in runs
        ],
    }


def _pick_best(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    def key_fn(r: Dict[str, Any]):
        t = r.get("test", {}) or {}
        return (float(t.get("f1", -1.0)), float(t.get("accuracy", -1.0)))

    return sorted(runs, key=key_fn, reverse=True)[0]


def _copytree_overwrite(src: str, dst: str) -> None:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "MINERVA Script 17: Train detectors across multiple random seeds (default=5) and "
            "write summary statistics. Also exports the best seed into legacy directories "
            "expected by Scripts 06/12/15."
        )
    )
    ap.add_argument("--run_id", required=True,
                    help="e.g., 20260225_colab_run1")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--splits_dir", default="data/processed")
    ap.add_argument("--output_base", default="models")
    ap.add_argument("--resume_from_checkpoint", default="auto")
    ap.add_argument(
        "--no_export_legacy",
        action="store_true",
        help="Do NOT copy best-seed checkpoints to models/roberta_finetuned and models/distilbert_multilingual_finetuned.",
    )
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    # 1) Train RoBERTa and DistilBERT per seed
    for seed in seeds:
        subprocess.check_call(
            [
                sys.executable,
                "scripts/04_train_robertaMINERVA.py",
                "--run_id",
                args.run_id,
                "--seed",
                str(seed),
                "--splits_dir",
                args.splits_dir,
                "--output_base",
                args.output_base,
                "--resume_from_checkpoint",
                args.resume_from_checkpoint,
            ]
        )
        subprocess.check_call(
            [
                sys.executable,
                "scripts/05_train_distilbertMINERVA.py",
                "--run_id",
                args.run_id,
                "--seed",
                str(seed),
                "--splits_dir",
                args.splits_dir,
                "--output_base",
                args.output_base,
                "--resume_from_checkpoint",
                args.resume_from_checkpoint,
            ]
        )

    # 2) Collect metrics.json per seed
    roberta_runs: List[Dict[str, Any]] = []
    distil_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        roberta_path = os.path.join(
            args.output_base, "roberta", f"run_{args.run_id}", f"seed_{seed}", "metrics.json"
        )
        distil_path = os.path.join(
            args.output_base, "distilbert", f"run_{args.run_id}", f"seed_{seed}", "metrics.json"
        )
        roberta_runs.append(_read_json(roberta_path))
        distil_runs.append(_read_json(distil_path))

    summary = {
        "run_id": args.run_id,
        "seeds": seeds,
        "roberta": _summarize("jcblaise/roberta-tagalog-base", roberta_runs),
        "distilbert": _summarize("distilbert-base-multilingual-cased", distil_runs),
    }

    _ensure_dir("reports")
    out_summary = os.path.join(
        "reports", f"detectors_5seed_summary_{args.run_id}.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[OK] Wrote summary ->", out_summary)

    # 3) Export best seed into legacy directories (so downstream scripts work out-of-the-box)
    if not args.no_export_legacy:
        best_roberta = _pick_best(roberta_runs)
        best_distil = _pick_best(distil_runs)

        legacy_roberta = os.path.join(args.output_base, "roberta_finetuned")
        legacy_distil = os.path.join(
            args.output_base, "distilbert_multilingual_finetuned")

        _copytree_overwrite(best_roberta["output_dir"], legacy_roberta)
        _copytree_overwrite(best_distil["output_dir"], legacy_distil)

        export_info = {
            "run_id": args.run_id,
            "best_roberta": {
                "seed": best_roberta.get("seed"),
                "src": best_roberta.get("output_dir"),
                "dst": legacy_roberta,
                "test": best_roberta.get("test"),
            },
            "best_distilbert": {
                "seed": best_distil.get("seed"),
                "src": best_distil.get("output_dir"),
                "dst": legacy_distil,
                "test": best_distil.get("test"),
            },
        }

        out_export = os.path.join(
            "reports", f"best_seed_export_{args.run_id}.json")
        with open(out_export, "w", encoding="utf-8") as f:
            json.dump(export_info, f, indent=2)

        print("[OK] Exported best detectors to legacy paths:")
        print(" -", legacy_roberta,
              "(seed:", export_info["best_roberta"]["seed"], ")")
        print(" -", legacy_distil,
              "(seed:", export_info["best_distilbert"]["seed"], ")")


if __name__ == "__main__":
    main()

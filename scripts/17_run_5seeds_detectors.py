import argparse
import json
import os
import statistics
import subprocess
import sys
from glob import glob


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_metrics(metrics_path: str) -> dict:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize(model_name: str, runs: list) -> dict:
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
                "seed": r["seed"],
                "output_dir": r["output_dir"],
                "test": r["test"],
            } for r in runs
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True,
                    help="e.g., 20260220_colab_run1")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--splits_dir", default="data/processed")
    ap.add_argument("--output_base", default="models")
    ap.add_argument("--resume_from_checkpoint", default="auto")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    for seed in seeds:
        subprocess.check_call([
            sys.executable, "scripts/04_train_robertaMINERVA.py",
            "--run_id", args.run_id,
            "--seed", str(seed),
            "--splits_dir", args.splits_dir,
            "--output_base", args.output_base,
            "--resume_from_checkpoint", args.resume_from_checkpoint
        ])
        subprocess.check_call([
            sys.executable, "scripts/05_train_distilbertMINERVA.py",
            "--run_id", args.run_id,
            "--seed", str(seed),
            "--splits_dir", args.splits_dir,
            "--output_base", args.output_base,
            "--resume_from_checkpoint", args.resume_from_checkpoint
        ])

    roberta_metrics = []
    distilbert_metrics = []

    for seed in seeds:
        roberta_path = os.path.join(
            args.output_base, "roberta", f"run_{args.run_id}", f"seed_{seed}", "metrics.json")
        distilbert_path = os.path.join(
            args.output_base, "distilbert", f"run_{args.run_id}", f"seed_{seed}", "metrics.json")
        roberta_metrics.append(_read_metrics(roberta_path))
        distilbert_metrics.append(_read_metrics(distilbert_path))

    summary = {
        "run_id": args.run_id,
        "seeds": seeds,
        "roberta": _summarize("jcblaise/roberta-tagalog-base", roberta_metrics),
        "distilbert": _summarize("distilbert-base-multilingual-cased", distilbert_metrics),
    }

    _ensure_dir("reports")
    out = os.path.join(
        "reports", f"detectors_5seed_summary_{args.run_id}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[OK] Wrote summary ->", out)


if __name__ == "__main__":
    main()

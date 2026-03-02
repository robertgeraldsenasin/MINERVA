#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _metrics_path(output_base: str, task: str, run_id: str, seed: int) -> Path:
    return Path(output_base) / task / f"run_{run_id}" / f"seed_{seed}" / "metrics.json"


def _seed_out_dir(output_base: str, task: str, run_id: str, seed: int) -> Path:
    return Path(output_base) / task / f"run_{run_id}" / f"seed_{seed}"


def _normalize_metrics_schema(r: Dict[str, Any]) -> Dict[str, Any]:

    # Ensure output_dir exists (Script 16 writes output_dir; just keep safe fallback)
    if "output_dir" not in r and "output_dir" not in r:
        # If missing, we can't export/copy later. Caller should provide.
        pass

    # Normalize test metrics
    if "test" not in r or not isinstance(r.get("test"), dict):
        tm = r.get("test_metrics")
        if isinstance(tm, dict):
            r["test"] = tm

    # Normalize val metrics (optional)
    if "val" not in r or not isinstance(r.get("val"), dict):
        em = r.get("eval_metrics")
        if isinstance(em, dict):
            val = {}
            # Script 16 stores eval_* keys under eval_metrics dict
            # but sometimes eval_metrics already is flattened; handle both.
            for k, v in em.items():
                if k.startswith("eval_"):
                    # map eval_f1 -> f1 etc.
                    name = k.replace("eval_", "")
                    try:
                        val[name] = float(v)
                    except Exception:
                        pass
            if val:
                r["val"] = val

    if "test" not in r or not isinstance(r["test"], dict):
        raise KeyError(
            "metrics.json missing required test metrics. "
            "Expected either 'test' or 'test_metrics' dict."
        )

    return r


def _summarize(model_name: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    test_f1 = [float(r["test"]["f1"]) for r in runs]
    test_acc = [float(r["test"]["accuracy"]) for r in runs]

    return {
        "model": model_name,
        "n_runs": len(runs),
        "test_f1_mean": float(statistics.mean(test_f1)),
        "test_f1_std": float(statistics.pstdev(test_f1)) if len(test_f1) > 1 else 0.0,
        "test_acc_mean": float(statistics.mean(test_acc)),
        "test_acc_std": float(statistics.pstdev(test_acc)) if len(test_acc) > 1 else 0.0,
        "per_seed": [
            {
                "seed": r.get("seed"),
                "output_dir": r.get("output_dir"),
                "test": r.get("test"),
                "val": r.get("val"),
            }
            for r in runs
        ],
    }


def _pick_best(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Best = highest test F1; tie-breaker = highest test accuracy.
    """
    def key_fn(r: Dict[str, Any]) -> Tuple[float, float]:
        t = r.get("test", {}) or {}
        return (float(t.get("f1", -1.0)), float(t.get("accuracy", -1.0)))

    return sorted(runs, key=key_fn, reverse=True)[0]


def _copytree_overwrite(src: str | Path, dst: str | Path) -> None:
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _run(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    repo_root = _repo_root()
    os.chdir(repo_root)

    ap = argparse.ArgumentParser(
        description=(
            "MINERVA Script 17: Train RoBERTa + DistilBERT detectors across multiple random seeds "
            "(default=5) and write summary statistics. Also exports the best seed into legacy "
            "directories expected by Scripts 06/12/15."
        )
    )
    ap.add_argument("--run_id", required=True,
                    help="e.g., 20260226_colab_run1")
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="Comma-separated seeds.")
    ap.add_argument("--splits_dir", default="data/processed")
    ap.add_argument("--output_base", default="models")
    ap.add_argument("--resume_from_checkpoint", default="auto",
                    help="auto | none | <path> (forwarded to Script 16)")
    ap.add_argument(
        "--no_export_legacy",
        action="store_true",
        help="Do NOT copy best-seed outputs to models/roberta_finetuned and models/distilbert_multilingual_finetuned.",
    )
    ap.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip training for a seed/task if metrics.json already exists.",
    )
    ap.add_argument(
        "--clean_seed_dirs",
        action="store_true",
        help="Delete existing models/<task>/run_<run_id>/seed_<seed> before training (forces fresh retrain).",
    )

    # IMPORTANT: allow forwarding extra args to Script 16 via wrappers 04/05
    args, extra = ap.parse_known_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("No seeds provided.")

    # 1) Train detectors per seed
    for seed in seeds:
        # Optional clean wipe for fresh retrain
        if args.clean_seed_dirs:
            for task in ("roberta", "distilbert"):
                d = _seed_out_dir(args.output_base, task, args.run_id, seed)
                if d.exists():
                    print(f"[CLEAN] Removing {d}")
                    shutil.rmtree(d)

        # Train RoBERTa
        roberta_metrics = _metrics_path(
            args.output_base, "roberta", args.run_id, seed)
        if args.skip_existing and roberta_metrics.exists():
            print(f"[SKIP] RoBERTa seed {seed} (exists): {roberta_metrics}")
        else:
            _run(
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
                + extra
            )

        # Train DistilBERT
        distil_metrics = _metrics_path(
            args.output_base, "distilbert", args.run_id, seed)
        if args.skip_existing and distil_metrics.exists():
            print(f"[SKIP] DistilBERT seed {seed} (exists): {distil_metrics}")
        else:
            _run(
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
                + extra
            )

    # 2) Collect metrics.json per seed + normalize schema
    roberta_runs: List[Dict[str, Any]] = []
    distil_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        roberta_path = _metrics_path(
            args.output_base, "roberta", args.run_id, seed)
        distil_path = _metrics_path(
            args.output_base, "distilbert", args.run_id, seed)

        if not roberta_path.exists():
            raise FileNotFoundError(
                f"Missing RoBERTa metrics.json: {roberta_path}")
        if not distil_path.exists():
            raise FileNotFoundError(
                f"Missing DistilBERT metrics.json: {distil_path}")

        rr = _normalize_metrics_schema(_read_json(roberta_path))
        dr = _normalize_metrics_schema(_read_json(distil_path))

        roberta_runs.append(rr)
        distil_runs.append(dr)

    summary = {
        "run_id": args.run_id,
        "seeds": seeds,
        "roberta": _summarize("jcblaise/roberta-tagalog-base", roberta_runs),
        "distilbert": _summarize("distilbert-base-multilingual-cased", distil_runs),
    }

    out_summary = Path("reports") / \
        f"detectors_5seed_summary_{args.run_id}.json"
    _write_json(out_summary, summary)
    print("[OK] Wrote summary ->", out_summary)

    # 3) Export best seed into legacy directories (so downstream scripts work out-of-the-box)
    if not args.no_export_legacy:
        best_roberta = _pick_best(roberta_runs)
        best_distil = _pick_best(distil_runs)

        if not best_roberta.get("output_dir") or not best_distil.get("output_dir"):
            raise KeyError(
                "metrics.json is missing 'output_dir'. "
                "Your Script 16 should write output_dir in metrics.json."
            )

        legacy_roberta = Path(args.output_base) / "roberta_finetuned"
        legacy_distil = Path(args.output_base) / \
            "distilbert_multilingual_finetuned"

        _copytree_overwrite(best_roberta["output_dir"], legacy_roberta)
        _copytree_overwrite(best_distil["output_dir"], legacy_distil)

        export_info = {
            "run_id": args.run_id,
            "best_roberta": {
                "seed": best_roberta.get("seed"),
                "src": best_roberta.get("output_dir"),
                "dst": str(legacy_roberta),
                "test": best_roberta.get("test"),
            },
            "best_distilbert": {
                "seed": best_distil.get("seed"),
                "src": best_distil.get("output_dir"),
                "dst": str(legacy_distil),
                "test": best_distil.get("test"),
            },
        }

        out_export = Path("reports") / f"best_seed_export_{args.run_id}.json"
        _write_json(out_export, export_info)

        print("[OK] Exported best detectors to legacy paths:")
        print(" -", legacy_roberta,
              "(seed:", export_info["best_roberta"]["seed"], ")")
        print(" -", legacy_distil,
              "(seed:", export_info["best_distilbert"]["seed"], ")")
        print("[OK] Wrote export info ->", out_export)

        # Quick sanity: verify legacy folders have config.json so Script 15 won't crash
        for cfg in (legacy_roberta / "config.json", legacy_distil / "config.json"):
            if not cfg.exists():
                print(
                    "[WARN] Missing", cfg,
                    "-> ensure Script 16 calls trainer.save_model(out_dir) and tokenizer.save_pretrained(out_dir)."
                )


if __name__ == "__main__":
    main()

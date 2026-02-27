from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _ckpt_step(path: str) -> int:
    """Extract checkpoint step from a path like '.../checkpoint-195'."""
    m = re.search(r"checkpoint-(\d+)", path)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON in {path}, got {type(obj)}")
    return obj


def _find_trainer_state(seed_dir: str) -> Optional[str]:
    """Locate a trainer_state-like file for this seed.

    Returns a JSON path that contains a dict with a 'log_history' list.
    """
    direct = os.path.join(seed_dir, "trainer_state.json")
    if os.path.exists(direct):
        return direct

    # Checkpoint fallback (latest)
    ckpt_states = glob(os.path.join(
        seed_dir, "checkpoint-*", "trainer_state.json"))
    if ckpt_states:
        ckpt_states = sorted(ckpt_states, key=_ckpt_step)
        return ckpt_states[-1]

    # metrics.json fallback (if it contains log_history)
    metrics = os.path.join(seed_dir, "metrics.json")
    if os.path.exists(metrics):
        try:
            m = _load_json(metrics)
            if isinstance(m.get("log_history"), list):
                return metrics
        except Exception:
            pass

    return None


def _extract_curves(
    state: Dict[str, Any],
) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float], List[Optional[float]]]]:
    """Return ((train_steps, train_loss), (eval_steps, eval_loss, eval_f1))."""
    hist = state.get("log_history", [])
    if not isinstance(hist, list):
        hist = []

    train_steps: List[float] = []
    train_loss: List[float] = []
    eval_steps: List[float] = []
    eval_loss: List[float] = []
    eval_f1: List[Optional[float]] = []

    # Some logs include only epoch; we build a monotonic fallback index.
    fallback_step = 0

    for row in hist:
        if not isinstance(row, dict):
            continue

        step = row.get("step")
        if step is None:
            step = row.get("global_step")
        if step is None:
            fallback_step += 1
            step = fallback_step

        # train loss rows generally have 'loss' and *no* 'eval_loss'
        if "loss" in row and "eval_loss" not in row:
            try:
                train_steps.append(float(step))
                train_loss.append(float(row["loss"]))
            except Exception:
                pass

        if "eval_loss" in row:
            try:
                eval_steps.append(float(step))
                eval_loss.append(float(row["eval_loss"]))
            except Exception:
                continue

            f1_val = row.get("eval_f1")
            if f1_val is None:
                eval_f1.append(None)
            else:
                try:
                    eval_f1.append(float(f1_val))
                except Exception:
                    eval_f1.append(None)

    return (train_steps, train_loss), (eval_steps, eval_loss, eval_f1)


def _plot_seed(task: str, run_id: str, seed_dir: str, out_dir: str) -> None:
    seed_name = os.path.basename(seed_dir)

    state_path = _find_trainer_state(seed_dir)
    if state_path is None:
        print(
            "[WARN] Missing trainer_state.json for seed dir (and no checkpoint/metrics fallback):",
            seed_dir,
        )
        return

    try:
        state = _load_json(state_path)
    except Exception as e:
        print(f"[WARN] Failed to read {state_path}: {e}")
        return

    if not isinstance(state.get("log_history"), list):
        print(
            f"[WARN] Found {state_path} but it has no log_history; skipping {seed_name}.")
        return

    (ts, tl), (es, el, ef1) = _extract_curves(state)

    # Loss plot
    if ts or es:
        plt.figure()
        if ts:
            plt.plot(ts, tl, label="train_loss")
        if es:
            plt.plot(es, el, label="eval_loss")
        plt.title(f"{task} {seed_name} loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        out_loss = os.path.join(
            out_dir, f"{task}_{run_id}_{seed_name}_loss.png")
        plt.savefig(out_loss, dpi=180, bbox_inches="tight")
        plt.close()

    # F1 plot
    if any(v is not None for v in ef1):
        plt.figure()
        plt.plot(es, [v if v is not None else 0.0 for v in ef1],
                 label="eval_f1")
        plt.title(f"{task} {seed_name} eval F1")
        plt.xlabel("step")
        plt.ylabel("F1")
        plt.legend()
        out_f1 = os.path.join(out_dir, f"{task}_{run_id}_{seed_name}_f1.png")
        plt.savefig(out_f1, dpi=180, bbox_inches="tight")
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_base", default="models")
    ap.add_argument("--run_id", required=True)
    ap.add_argument(
        "--task", choices=["roberta", "distilbert", "all"], default="all")
    ap.add_argument("--out_dir", default="reports/figures")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    tasks = ["roberta", "distilbert"] if args.task == "all" else [args.task]

    any_plotted = False

    for task in tasks:
        run_dir = os.path.join(args.models_base, task, f"run_{args.run_id}")
        seed_dirs = sorted(glob(os.path.join(run_dir, "seed_*")))
        if not seed_dirs:
            print(
                f"[WARN] No seed dirs found for task={task} run_id={args.run_id} at: {run_dir}")
            continue

        for sd in seed_dirs:
            _plot_seed(task, args.run_id, sd, args.out_dir)
            any_plotted = True

    if any_plotted:
        print("[OK] Wrote plots to:", args.out_dir)
    else:
        print(
            "[WARN] No plots were generated. Either the run_id/task is wrong, or your detector runs don't have\n"
            "       trainer_state.json (and no checkpoint-*/trainer_state.json). If you need training curves,\n"
            "       re-run script 17 (or script 16) *without deleting checkpoints* (or explicitly call\n"
            "       trainer.save_state() after training)."
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import matplotlib.pyplot as plt


def _load_trainer_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_curves(state: dict):
    hist = state.get("log_history", [])
    train_steps, train_loss = [], []
    eval_steps, eval_loss, eval_f1 = [], [], []

    for row in hist:
        if "loss" in row and "step" in row and "eval_loss" not in row:
            train_steps.append(row["step"])
            train_loss.append(row["loss"])
        if "eval_loss" in row and "step" in row:
            eval_steps.append(row["step"])
            eval_loss.append(row["eval_loss"])
            if "eval_f1" in row:
                eval_f1.append(row["eval_f1"])
            else:
                eval_f1.append(None)

    return (train_steps, train_loss), (eval_steps, eval_loss, eval_f1)


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_base", default="models")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--task", choices=["roberta", "distilbert"], required=True)
    ap.add_argument("--out_dir", default="reports/figures")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    seed_dirs = sorted(
        glob(os.path.join(args.models_base, args.task, f"run_{args.run_id}", "seed_*")))
    if not seed_dirs:
        raise FileNotFoundError("No seed dirs found.")

    for sd in seed_dirs:
        state_path = os.path.join(sd, "trainer_state.json")
        if not os.path.exists(state_path):
            # Trainer always writes it; if missing, training didn’t complete
            print("[WARN] Missing trainer_state.json:", state_path)
            continue

        state = _load_trainer_state(state_path)
        (ts, tl), (es, el, ef1) = _extract_curves(state)

        seed_name = os.path.basename(sd)
        # Loss plot
        plt.figure()
        if ts:
            plt.plot(ts, tl, label="train_loss")
        if es:
            plt.plot(es, el, label="eval_loss")
        plt.title(f"{args.task} {seed_name} loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend()
        out_loss = os.path.join(
            args.out_dir, f"{args.task}_{args.run_id}_{seed_name}_loss.png")
        plt.savefig(out_loss, dpi=180, bbox_inches="tight")
        plt.close()

        # F1 plot
        if any(x is not None for x in ef1):
            plt.figure()
            plt.plot(
                es, [x if x is not None else 0.0 for x in ef1], label="eval_f1")
            plt.title(f"{args.task} {seed_name} eval F1")
            plt.xlabel("step")
            plt.ylabel("F1")
            plt.legend()
            out_f1 = os.path.join(
                args.out_dir, f"{args.task}_{args.run_id}_{seed_name}_f1.png")
            plt.savefig(out_f1, dpi=180, bbox_inches="tight")
            plt.close()

    print("[OK] Wrote plots to:", args.out_dir)


if __name__ == "__main__":
    main()

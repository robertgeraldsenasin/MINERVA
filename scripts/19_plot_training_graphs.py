from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _checkpoint_step_name(p: Path) -> int:
    """Return the numeric step suffix from 'checkpoint-<N>' or -1."""
    try:
        return int(p.name.split("-")[-1])
    except Exception:
        return -1


def _find_trainer_state(seed_dir: Path) -> Optional[Path]:
    """Find trainer_state.json for a seed directory.

    Preferred order:
      1) <seed_dir>/trainer_state.json
      2) latest <seed_dir>/checkpoint-*/trainer_state.json
      3) any nested trainer_state.json (rglob fallback)
    """
    direct = seed_dir / "trainer_state.json"
    if direct.exists():
        return direct

    # Hugging Face Trainer writes trainer_state.json inside checkpoint folders.
    ckpts = sorted(
        [p for p in seed_dir.glob("checkpoint-*") if p.is_dir()],
        key=_checkpoint_step_name,
        reverse=True,
    )
    for c in ckpts:
        p = c / "trainer_state.json"
        if p.exists():
            return p

    # Last resort: search recursively (should be rare; can be slow on huge dirs)
    for p in seed_dir.rglob("trainer_state.json"):
        return p

    return None


def _extract_curves(state: dict) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float], List[Optional[float]]], str]:
    """Extract train/eval curves from HF TrainerState.

    Returns:
        (train_x, train_loss), (eval_x, eval_loss, eval_f1), x_label
    """
    hist = state.get("log_history", []) or []

    train_x, train_loss = [], []
    eval_x, eval_loss, eval_f1 = [], [], []

    # Determine whether "step" exists; if not, fall back to "epoch".
    has_step = any(isinstance(r, dict) and "step" in r for r in hist)
    x_key = "step" if has_step else "epoch"

    for row in hist:
        if not isinstance(row, dict):
            continue

        x = row.get(x_key)
        if x is None:
            # fall back to whichever exists
            x = row.get("step", row.get("epoch"))
        if x is None:
            continue

        # training loss entries have "loss" and *not* "eval_loss"
        if "loss" in row and "eval_loss" not in row:
            try:
                train_x.append(float(x))
                train_loss.append(float(row["loss"]))
            except Exception:
                pass

        # eval entries have "eval_loss"
        if "eval_loss" in row:
            try:
                eval_x.append(float(x))
                eval_loss.append(float(row["eval_loss"]))
                eval_f1.append(float(row["eval_f1"]) if "eval_f1" in row and row["eval_f1"] is not None else None)
            except Exception:
                pass

    return (train_x, train_loss), (eval_x, eval_loss, eval_f1), x_key


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "MINERVA Script 19: Plot training curves (loss/F1) for each seed.\n"
            "Robust to trainer_state.json being stored inside checkpoint directories."
        )
    )
    ap.add_argument("--models_base", default="models")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--task", choices=["roberta", "distilbert"], required=True)
    ap.add_argument("--out_dir", default="reports/figures")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    seed_dirs = sorted(
        glob(os.path.join(args.models_base, args.task, f"run_{args.run_id}", "seed_*"))
    )
    if not seed_dirs:
        raise FileNotFoundError(
            f"No seed dirs found for task='{args.task}' run_id='{args.run_id}' under {args.models_base}"
        )

    wrote = 0
    missing = 0

    for sd_str in seed_dirs:
        sd = Path(sd_str)
        state_path = _find_trainer_state(sd)
        if not state_path:
            print("[WARN] Missing trainer_state.json (seed + checkpoints):", sd)
            missing += 1
            continue

        try:
            state = _load_json(state_path)
        except Exception as e:
            print("[WARN] Failed to read trainer_state.json:", state_path, "err:", repr(e))
            missing += 1
            continue

        (tx, tl), (ex, el, ef1), x_key = _extract_curves(state)

        seed_name = sd.name
        x_label = "step" if x_key == "step" else "epoch"

        # Loss plot
        if tx or ex:
            plt.figure()
            if tx:
                plt.plot(tx, tl, label="train_loss")
            if ex:
                plt.plot(ex, el, label="eval_loss")
            plt.title(f"{args.task} {seed_name} loss")
            plt.xlabel(x_label)
            plt.ylabel("loss")
            plt.legend()
            out_loss = os.path.join(args.out_dir, f"{args.task}_{args.run_id}_{seed_name}_loss.png")
            plt.savefig(out_loss, dpi=180, bbox_inches="tight")
            plt.close()
            wrote += 1
        else:
            print("[WARN] No loss curve data found in:", state_path)

        # F1 plot (skip None points)
        f1_points = [(x, f) for x, f in zip(ex, ef1) if f is not None]
        if f1_points:
            fx = [p[0] for p in f1_points]
            ff = [p[1] for p in f1_points]
            plt.figure()
            plt.plot(fx, ff, label="eval_f1")
            plt.title(f"{args.task} {seed_name} eval F1")
            plt.xlabel(x_label)
            plt.ylabel("F1")
            plt.legend()
            out_f1 = os.path.join(args.out_dir, f"{args.task}_{args.run_id}_{seed_name}_f1.png")
            plt.savefig(out_f1, dpi=180, bbox_inches="tight")
            plt.close()
            wrote += 1

    if wrote == 0:
        print(
            "[WARN] No plots were written. This usually means trainer_state.json is missing "
            "everywhere (seed dir + checkpoints) or contains no log_history."
        )
    else:
        print(f"[OK] Wrote {wrote} plot(s) to: {args.out_dir}")

    if missing:
        print(f"[INFO] Seeds missing trainer_state/log_history: {missing} / {len(seed_dirs)}")


if __name__ == "__main__":
    main()

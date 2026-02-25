from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    cmd = [
        sys.executable,
        "scripts/16_train_transformer_classifier.py",
        "--task", "distilbert",
        "--model_name", "distilbert-base-multilingual-cased",
    ] + sys.argv[1:]

    print("[WRAPPER 05] Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

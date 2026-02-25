from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    # Ensure we run from repo root even if called from elsewhere
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    cmd = [
        sys.executable,
        "scripts/16_train_transformer_classifier.py",
        "--task", "roberta",
        "--model_name", "jcblaise/roberta-tagalog-base",
    ] + sys.argv[1:]  # forward all args from Script 17

    print("[WRAPPER 04] Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "experiment" / "model_optimization"
RUNNER = WORK_DIR / "run_optuna_family.py"
STDOUT = WORK_DIR / "search_queue.stdout.log"
STDERR = WORK_DIR / "search_queue.stderr.log"

COMMANDS = [
    [
        sys.executable, str(RUNNER), "--family", "xgboost", "--trials", "140",
        "--folds", "2023,2024", "--study-name", "xgboost_v1_full_2023_2024",
    ],
    [
        sys.executable, str(RUNNER), "--family", "catboost", "--trials", "140",
        "--folds", "2023,2024", "--study-name", "catboost_v1_full_2023_2024",
    ],
]


with STDOUT.open("a", encoding="utf-8", buffering=1) as stdout, STDERR.open(
    "a", encoding="utf-8", buffering=1
) as stderr:
    for command in COMMANDS:
        stdout.write(f"[{datetime.now().isoformat()}] START {' '.join(command)}\n")
        result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        stdout.write(
            f"[{datetime.now().isoformat()}] END returncode={result.returncode} "
            f"command={' '.join(command)}\n"
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    stdout.write(f"[{datetime.now().isoformat()}] SEARCH QUEUE COMPLETE\n")


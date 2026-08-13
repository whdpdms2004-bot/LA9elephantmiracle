from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent
REQUIRED = [
    WORK_DIR / "xgboost_v1_full_2023_2024_summary.json",
    WORK_DIR / "catboost_v1_full_2023_2024_summary.json",
]
LOG = WORK_DIR / "ensemble_queue.log"
DEADLINE = time.time() + 18 * 60 * 60

with LOG.open("a", encoding="utf-8", buffering=1) as log:
    log.write("Waiting for Optuna summaries.\n")
    while time.time() < DEADLINE and not all(path.is_file() for path in REQUIRED):
        time.sleep(60)
    if not all(path.is_file() for path in REQUIRED):
        log.write("Timed out waiting for Optuna summaries.\n")
        raise SystemExit(2)
    log.write("Optuna summaries found. Building OOF ensemble.\n")
    result = subprocess.run(
        [sys.executable, str(WORK_DIR / "build_oof_ensemble.py")],
        cwd=WORK_DIR.parents[1], stdout=log, stderr=log, check=False,
    )
    log.write(f"OOF ensemble returncode={result.returncode}\n")
    raise SystemExit(result.returncode)


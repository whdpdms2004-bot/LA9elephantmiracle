from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "experiment" / "model_optimization"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["xgboost", "catboost"], required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--study-name", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    stem = args.study_name
    stdout_path = WORK_DIR / f"{stem}.out.log"
    stderr_path = WORK_DIR / f"{stem}.err.log"
    command = [
        sys.executable,
        str(WORK_DIR / "run_optuna_enhanced.py"),
        "--family",
        args.family,
        "--trials",
        str(args.trials),
        "--folds",
        args.folds,
        "--study-name",
        args.study_name,
    ]
    environment = os.environ.copy()
    # Windows treats these keys as the same while Python's dict does not.
    # Keep one spelling so CreateProcess receives a valid environment block.
    path_value = environment.get("Path") or environment.get("PATH") or ""
    environment.pop("Path", None)
    environment.pop("PATH", None)
    environment["Path"] = path_value

    # CREATE_NO_WINDOW keeps the child alive without attaching a console.
    # DETACHED_PROCESS is intentionally omitted: Intel/Conda DLLs can fail to
    # initialize under that flag on Windows.
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creation_flags,
            close_fds=True,
        )

    state = {
        "pid": process.pid,
        "started_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "cwd": str(ROOT),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "study_database": str(WORK_DIR / f"{args.study_name}.db"),
    }
    state_path = WORK_DIR / f"{stem}.process.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

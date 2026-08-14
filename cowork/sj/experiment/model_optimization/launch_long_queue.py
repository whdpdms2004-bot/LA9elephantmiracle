from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "experiment" / "model_optimization"


def main():
    environment = os.environ.copy()
    path_value = environment.get("Path") or environment.get("PATH") or ""
    environment.pop("Path", None)
    environment.pop("PATH", None)
    environment["Path"] = path_value
    stdout_path = WORK_DIR / "long_experiment_queue.out.log"
    stderr_path = WORK_DIR / "long_experiment_queue.err.log"
    command = [sys.executable, str(WORK_DIR / "continue_long_experiments.py")]
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
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            ),
            close_fds=True,
        )
    state = {
        "launcher_pid": process.pid,
        "started_at": datetime.now().astimezone().isoformat(),
        "command": command,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    (WORK_DIR / "long_experiment_queue.process.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

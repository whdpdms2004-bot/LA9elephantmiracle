from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from run_optuna_family import ROOT


WORK = ROOT / "experiment" / "model_optimization"
OUTPUT = WORK / "failure_experts"
STATUS_PATH = OUTPUT / "campaign_status.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    return parser.parse_args()


def process_exists(pid: int) -> bool:
    process = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not process:
        return False
    exit_code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(process, ctypes.byref(exit_code))
    ctypes.windll.kernel32.CloseHandle(process)
    return exit_code.value == 259


def write_status(state, current_step, completed, error=None):
    payload = {
        "updated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "state": state,
        "current_step": current_step,
        "completed_steps": completed,
        "error": error,
    }
    STATUS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_step(name, arguments, completed):
    write_status("running", name, completed)
    stdout_path = OUTPUT / f"campaign_{name}.out.log"
    stderr_path = OUTPUT / f"campaign_{name}.err.log"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if result.returncode != 0:
        message = f"{name} failed with return code {result.returncode}"
        write_status("failed", name, completed, message)
        raise RuntimeError(message)
    completed.append(name)
    write_status("running", "transition", completed)


def main():
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = []
    write_status("waiting", "middle_search", completed)
    while process_exists(args.wait_pid):
        time.sleep(30)

    steps = [
        (
            "middle_gate",
            [
                "experiment/model_optimization/evaluate_failure_expert_top.py",
                "--head",
                "middle",
                "--top-count",
                "8",
            ],
        ),
        (
            "reverse_search",
            [
                "experiment/model_optimization/run_optuna_failure_expert.py",
                "--head",
                "reverse",
                "--target-total",
                "120",
            ],
        ),
        (
            "reverse_gate",
            [
                "experiment/model_optimization/evaluate_failure_expert_top.py",
                "--head",
                "reverse",
                "--top-count",
                "8",
            ],
        ),
        (
            "outside_search",
            [
                "experiment/model_optimization/run_optuna_failure_expert.py",
                "--head",
                "outside",
                "--target-total",
                "100",
            ],
        ),
        (
            "outside_gate",
            [
                "experiment/model_optimization/evaluate_failure_expert_top.py",
                "--head",
                "outside",
                "--top-count",
                "8",
            ],
        ),
    ]
    for name, arguments in steps:
        run_step(name, arguments, completed)
    write_status("complete", None, completed)


if __name__ == "__main__":
    main()

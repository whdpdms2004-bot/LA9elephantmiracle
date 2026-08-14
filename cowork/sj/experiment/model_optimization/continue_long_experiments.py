from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK_DIR = ROOT / "experiment" / "model_optimization"
STATE_PATH = WORK_DIR / "long_experiment_queue.json"
WAIT_PID = 38876


def now():
    return datetime.now().astimezone().isoformat()


def write_state(state):
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def initialize_state(queue):
    previous = None
    if STATE_PATH.is_file():
        try:
            previous = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = None
    completed = set()
    original_created_at = now()
    if previous:
        original_created_at = previous.get("created_at", original_created_at)
        completed = {
            item["name"]
            for item in previous.get("steps", [])
            if item.get("status") == "completed"
        }
    return {
        "created_at": original_created_at,
        "resumed_at": now(),
        "wait_pid": WAIT_PID,
        "status": "waiting_for_robust_xgb",
        "steps": [
            {
                "name": item["name"],
                "status": "completed" if item["name"] in completed else "pending",
                "command": item["command"],
            }
            for item in queue
        ],
    }


def wait_for_process(pid: int):
    synchronize = 0x00100000
    handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        return
    try:
        ctypes.windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def normalized_environment():
    environment = os.environ.copy()
    path_value = environment.get("Path") or environment.get("PATH") or ""
    environment.pop("Path", None)
    environment.pop("PATH", None)
    environment["Path"] = path_value
    return environment


def commands():
    python = sys.executable
    registry = [python, str(WORK_DIR / "build_validation_registry.py")]
    embedding_script = (
        ROOT
        / "experiment"
        / "pitcher_embedding"
        / "train_strict_multitask_embedding.py"
    )
    return [
        {
            "name": "trackman_pitchgroup_fixed_ablation",
            "command": [
                python,
                str(WORK_DIR / "benchmark_trackman_pitchgroup_fixed.py"),
            ],
        },
        {"name": "registry_after_pitchgroup", "command": registry},
        {
            "name": "trackman_gated_ablation",
            "command": [python, str(WORK_DIR / "benchmark_trackman_gated.py")],
        },
        {"name": "registry_after_gated", "command": registry},
        {
            "name": "smoothing_grid_ablation",
            "command": [python, str(WORK_DIR / "benchmark_smoothing_grid.py")],
        },
        {"name": "registry_after_smoothing", "command": registry},
        {
            "name": "direct_brier_ablation",
            "command": [python, str(WORK_DIR / "benchmark_direct_brier.py")],
        },
        {"name": "registry_after_direct_brier", "command": registry},
        {
            "name": "xgb_enhanced_recent_120",
            "command": [
                python,
                str(WORK_DIR / "run_optuna_enhanced.py"),
                "--family",
                "xgboost",
                "--trials",
                "120",
                "--target-total",
                "120",
                "--folds",
                "2024",
                "--study-name",
                "xgboost_v2r200_tm500_2024",
            ],
        },
        {"name": "registry_after_xgb_recent", "command": registry},
        {
            "name": "xgb_enhanced_local_100",
            "command": [
                python,
                str(WORK_DIR / "run_optuna_xgb_local.py"),
                "--target-total",
                "100",
            ],
        },
        {"name": "registry_after_xgb_local", "command": registry},
        {
            "name": "lightgbm_enhanced_robust_60",
            "command": [
                python,
                str(WORK_DIR / "run_optuna_lightgbm_enhanced.py"),
                "--trials",
                "60",
                "--target-total",
                "60",
                "--folds",
                "2023,2024",
                "--study-name",
                "lightgbm_v2r200_tm500_robust",
            ],
        },
        {"name": "registry_after_lightgbm", "command": registry},
        {
            "name": "cat_enhanced_robust_80",
            "command": [
                python,
                str(WORK_DIR / "run_optuna_enhanced.py"),
                "--family",
                "catboost",
                "--trials",
                "80",
                "--target-total",
                "80",
                "--folds",
                "2023,2024",
                "--study-name",
                "catboost_v2r200_tm500_robust",
            ],
        },
        {"name": "registry_after_cat", "command": registry},
        *[
            {
                "name": f"strict_embedding_dim{dimension}",
                "command": [
                    python,
                    str(embedding_script),
                    "--embedding-dim",
                    str(dimension),
                    "--folds",
                    "2022,2023,2024",
                    "--epochs",
                    "8",
                    "--batch-size",
                    "8192",
                ],
            }
            for dimension in [16, 32, 64]
        ],
        {"name": "registry_after_embeddings", "command": registry},
        {
            "name": "enhanced_seed_oof",
            "command": [python, str(WORK_DIR / "build_enhanced_seed_oof.py")],
        },
        {"name": "registry_after_enhanced_oof", "command": registry},
        {
            "name": "optimize_enhanced_oof",
            "command": [python, str(WORK_DIR / "optimize_enhanced_oof.py")],
        },
        {"name": "registry_after_enhanced_ensemble", "command": registry},
        {
            "name": "build_final_two_submissions",
            "command": [python, str(WORK_DIR / "build_final_two_submissions.py")],
        },
    ]


def main():
    queue = commands()
    state = initialize_state(queue)
    write_state(state)
    wait_for_process(WAIT_PID)
    state["status"] = "running"
    state["robust_xgb_finished_at"] = now()
    write_state(state)
    environment = normalized_environment()

    for index, item in enumerate(queue):
        step = state["steps"][index]
        if step["status"] == "completed":
            continue
        step["status"] = "running"
        step["started_at"] = now()
        write_state(state)
        stdout_path = WORK_DIR / f"queue_{item['name']}.out.log"
        stderr_path = WORK_DIR / f"queue_{item['name']}.err.log"
        with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr_handle:
            result = subprocess.run(
                item["command"],
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
        step["return_code"] = result.returncode
        step["finished_at"] = now()
        step["stdout"] = str(stdout_path)
        step["stderr"] = str(stderr_path)
        step["status"] = "completed" if result.returncode == 0 else "failed"
        write_state(state)
        if result.returncode != 0:
            state["status"] = "failed"
            state["failed_step"] = item["name"]
            state["failed_at"] = now()
            write_state(state)
            return

    state["status"] = "completed"
    state["finished_at"] = now()
    write_state(state)


if __name__ == "__main__":
    main()

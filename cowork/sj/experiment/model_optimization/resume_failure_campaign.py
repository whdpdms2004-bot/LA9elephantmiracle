"""Safely resume the interrupted outside-failure campaign and run its gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import optuna
import pandas as pd

from run_optuna_family import ROOT


WORK = ROOT / "experiment" / "model_optimization"
OUTPUT = WORK / "failure_experts"
STATUS = OUTPUT / "campaign_status.json"
PID_PATH = OUTPUT / "resume_campaign.pid"
COMPLETED_PREFIX = ["middle_gate", "reverse_search", "reverse_gate"]


def write_status(state: str, step: str | None, completed: list[str], error=None):
    STATUS.write_text(
        json.dumps(
            {
                "updated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                "state": state,
                "current_step": step,
                "completed_steps": completed,
                "error": error,
                "resume_pid": os.getpid(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def clean_stale_trials() -> list[int]:
    study = optuna.load_study(
        study_name="xgb_failure_outside_robust",
        storage=f"sqlite:///{(OUTPUT / 'xgb_failure_outside_robust.db').as_posix()}",
    )
    stale = [
        trial.number
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.RUNNING
    ]
    for trial_number in stale:
        study.tell(
            trial_number,
            state=optuna.trial.TrialState.FAIL,
            skip_if_finished=True,
        )
    return stale


def run_logged(name: str, arguments: list[str]) -> None:
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
        raise RuntimeError(f"{name} failed with return code {result.returncode}")


def validate_search() -> dict:
    path = OUTPUT / "xgb_outside_status.json"
    status = json.loads(path.read_text(encoding="utf-8"))
    if int(status["attempted_trials"]) < 100:
        raise RuntimeError(f"outside search incomplete: {status['attempted_trials']}/100")
    if int(status["complete_trials"]) < 50:
        raise RuntimeError(f"too few complete trials: {status['complete_trials']}")
    return status


def validate_gate() -> dict:
    path = OUTPUT / "xgb_outside_gate_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    metrics = OUTPUT / "xgb_outside_gate_metrics.csv"
    oof = OUTPUT / "xgb_outside_gate_oof.parquet"
    if not metrics.is_file() or not oof.is_file():
        raise FileNotFoundError("outside gate metrics or OOF missing")
    return summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    completed = list(COMPLETED_PREFIX)
    try:
        stale = clean_stale_trials()
        write_status(
            "running",
            "outside_search",
            completed,
            error={"cleaned_stale_running_trials": stale},
        )
        run_logged(
            "outside_search_resume",
            [
                "experiment/model_optimization/run_optuna_failure_expert.py",
                "--head",
                "outside",
                "--target-total",
                "100",
            ],
        )
        search_status = validate_search()
        completed.append("outside_search")
        write_status("running", "outside_gate", completed)
        run_logged(
            "outside_gate_resume",
            [
                "experiment/model_optimization/evaluate_failure_expert_top.py",
                "--head",
                "outside",
                "--top-count",
                "8",
            ],
        )
        gate_summary = validate_gate()
        completed.append("outside_gate")
        write_status(
            "complete",
            None,
            completed,
            error={
                "cleaned_stale_running_trials": stale,
                "outside_search": search_status,
                "outside_gate": gate_summary,
            },
        )
    except Exception as exc:
        write_status(
            "failed",
            "outside_resume",
            completed,
            error={
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()

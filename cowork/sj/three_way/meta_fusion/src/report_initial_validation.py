"""submit_038 선택 규칙의 strict 2023 -> 2024 상세 검증 리포트."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_initial_fusion import (
    META, SUCCESS, bounds, fit_event_calibrator, load_fold, load_labeled,
    apply_event_calibrator, success_from_union,
)


def subset(y: np.ndarray, p: np.ndarray) -> dict:
    rate = float(y.mean())
    null = rate * (1.0 - rate)
    brier = float(np.mean((p - y) ** 2))
    return {
        "n": int(len(y)), "brier": brier,
        "bss": 100000.0 * (1.0 - brier / null),
        "target_mean": rate, "prediction_mean": float(p.mean()),
    }


def main() -> None:
    frame = load_labeled()
    train = load_fold(frame, 2023)
    valid = load_fold(frame, 2024)
    train_mask = ((frame["season"].to_numpy() == 2023)
                  & (frame["label_ok"].to_numpy() == 1))
    target_columns = {
        "middle": "y_middle", "reverse": "y_reverse",
        "outside": "y_outside", "mr": "y_mr",
    }
    calibrators = {
        key: fit_event_calibrator(
            train[key], frame.loc[train_mask, column].to_numpy(np.float64), 1e-4)
        for key, column in target_columns.items()
    }
    calibrated = {
        key: apply_event_calibrator(calibrators[key], valid[key], 0.25)
        for key in target_columns
    }
    calibrated["mr"] = np.clip(
        calibrated["mr"],
        *bounds(calibrated["middle"], calibrated["reverse"]))
    prediction = success_from_union(
        calibrated["middle"], calibrated["reverse"],
        calibrated["outside"], calibrated["mr"])
    baseline = success_from_union(
        valid["middle"], valid["reverse"], valid["outside"], valid["mr"])
    y = valid["y_success"]

    report = {
        "protocol": "fit calibrators on strict OOF2023, evaluate OOF2024",
        "l2": 1e-4,
        "strength": 0.25,
        "overall": subset(y, prediction),
        "baseline": subset(y, baseline),
        "delta_brier": float(
            np.mean((prediction - y) ** 2) - np.mean((baseline - y) ** 2)),
        "by_game_type": {},
        "by_month": {},
        "component_prediction_means": {
            key: {"before": float(valid[key].mean()),
                  "after": float(calibrated[key].mean())}
            for key in target_columns
        },
    }
    for game_type in ("R", "F"):
        mask = valid["game_type"] == game_type
        report["by_game_type"][game_type] = {
            "candidate": subset(y[mask], prediction[mask]),
            "baseline": subset(y[mask], baseline[mask]),
        }
    for month in sorted(np.unique(valid["game_month"]).tolist()):
        mask = valid["game_month"] == month
        report["by_month"][str(int(month))] = {
            "n": int(mask.sum()),
            "candidate_brier": float(np.mean((prediction[mask] - y[mask]) ** 2)),
            "baseline_brier": float(np.mean((baseline[mask] - y[mask]) ** 2)),
        }
    path = META / "outputs" / "initial_validation_report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

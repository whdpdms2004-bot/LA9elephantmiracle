"""F1 CatBoost의 엄격 순방향 3-fold 제출 점검 지표를 기록한다.

각 fold의 보정값은 해당 검증 시즌 이전 Target 시즌 평균만 사용한 고정 규칙으로
계산한다. 검증 예측 평균이나 검증 라벨 평균을 보정값 산출에 사용하지 않는다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "claude" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))

from evaluate_train_only_season_offsets import forecast_offset
from evaluate_bucketed_residual import logit, sigmoid
from harness import TARGET, load, metrics


PREDICTION_DIR = HERE / "outputs" / "single_catboost"
OUTPUT = HERE / "outputs" / "combined" / "f1_cat_validation_report.json"


def main() -> None:
    frame = load()
    rates = frame.groupby("season")[TARGET].mean()
    reports = {}
    combined_y = []
    combined_p = []
    combined_game_type = []
    combined_month = []

    for fold in (2022, 2023, 2024):
        valid = frame["season"].eq(fold).to_numpy()
        y = frame.loc[valid, TARGET].to_numpy(np.float64)
        prediction = np.load(PREDICTION_DIR / f"F1_{fold}.npy")
        offset = forecast_offset(rates, fold, window=None, damping=0.25)
        adjusted = sigmoid(logit(prediction) + offset)
        report = metrics(
            y,
            adjusted,
            game_type=frame.loc[valid, "game_type"].to_numpy(),
            month=frame.loc[valid, "game_month"].to_numpy(),
        )
        report["season_logit_offset"] = float(offset)
        report["month_brier"] = json.loads(report["month_brier"])
        reports[str(fold)] = report
        combined_y.append(y)
        combined_p.append(adjusted)
        combined_game_type.append(frame.loc[valid, "game_type"].to_numpy())
        combined_month.append(frame.loc[valid, "game_month"].to_numpy())

    aggregate = metrics(
        np.concatenate(combined_y),
        np.concatenate(combined_p),
        game_type=np.concatenate(combined_game_type),
        month=np.concatenate(combined_month),
    )
    aggregate["month_brier"] = json.loads(aggregate["month_brier"])
    reports["aggregate_2022_2024"] = aggregate
    reports["simple_fold_mean_bss_raw"] = float(np.mean([
        reports[str(fold)]["bss_raw"] for fold in (2022, 2023, 2024)
    ]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print(f"saved -> {OUTPUT}")


if __name__ == "__main__":
    main()

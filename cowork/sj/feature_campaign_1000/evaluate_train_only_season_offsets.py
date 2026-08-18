"""이전 시즌 Target 평균만으로 정한 고정 logit offset을 단독 C1에 검증한다.

예측값의 평균이나 검증/test 분포는 offset 계산에 사용하지 않는다.
각 fold S의 offset은 season < S인 Target의 시즌별 평균만으로 미리 고정한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_bucketed_residual import EPS, OUT, load, logit, prediction_path, sigmoid
from harness import TARGET, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["xgboost", "catboost"],
                        default="xgboost")
    parser.add_argument("--arms", default="B0,C1")
    return parser.parse_args()


def forecast_offset(rates: pd.Series, fold: int, window: int | None,
                    damping: float) -> float:
    history = rates.loc[rates.index < fold]
    if window is not None:
        history = history.iloc[-window:]
    x = history.index.to_numpy(np.float64)
    y = logit(history.to_numpy(np.float64))
    if len(history) < 2:
        return 0.0
    slope, intercept = np.polyfit(x, y, 1)
    forecast = intercept + slope * fold
    last = float(y[-1])
    return float(damping * (forecast - last))


def main() -> None:
    args = parse_args()
    arms = [value for value in args.arms.split(",") if value]
    df = load()
    rates = df.groupby("season")[TARGET].mean()
    specs = [("none", None, 0.0)]
    for window in (None, 3, 4, 5):
        for damping in (0.25, 0.50, 0.75, 1.00):
            tag = "all" if window is None else f"last{window}"
            specs.append((f"{tag}_d{int(damping * 100):03d}", window, damping))

    rows = []
    for fold in (2022, 2023, 2024):
        valid = df["season"].eq(fold).to_numpy()
        y = df.loc[valid, TARGET].to_numpy(np.float64)
        for arm in arms:
            if args.family == "xgboost":
                if arm in ("F0", "F1"):
                    pred = np.load(
                        HERE / "outputs" / "single_xgb"
                        / ("confirm_xgboost_v2r200_tm500_robust_cuda_efull_"
                           f"s20260818_{arm}_{fold}.npy"))
                else:
                    pred = np.load(prediction_path(arm, fold))
            else:
                pred = np.load(
                    HERE / "outputs" / "single_catboost" / f"{arm}_{fold}.npy")
            for name, window, damping in specs:
                offset = forecast_offset(rates, fold, window, damping)
                adjusted = sigmoid(logit(pred) + offset)
                score = metrics(y, adjusted)
                rows.append({
                    "fold": fold,
                    "arm": arm,
                    "method": name,
                    "window": window,
                    "damping": damping,
                    "offset": offset,
                    "bss_raw": score["bss_raw"],
                    "bss_centered": score["bss_centered"],
                    "brier": score["brier"],
                    "pred_mean": score["pred_mean"],
                })

    result = pd.DataFrame(rows)
    summary = []
    for (arm, method), group in result.groupby(["arm", "method"]):
        values = group.set_index("fold")["bss_raw"].to_dict()
        summary.append({
            "arm": arm,
            "method": method,
            **{str(fold): float(values[fold]) for fold in (2022, 2023, 2024)},
            "worst": float(min(values.values())),
            "mean": float(np.mean(list(values.values()))),
        })
    summary.sort(key=lambda row: (row["worst"], row["mean"]), reverse=True)
    result.to_csv(
        OUT / f"train_only_season_offsets_{args.family}.csv", index=False)
    path = OUT / f"train_only_season_offsets_{args.family}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).groupby("arm", group_keys=False).head(8).to_string(index=False))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()

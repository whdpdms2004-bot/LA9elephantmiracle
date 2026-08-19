"""캐시된 3WAY 순방향 검증 예측의 Brier/BSS/R·F/월별 지표를 기록한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from combine import load_pred
from harness3 import OUT, SUCCESS, bss, load_labeled


def subset_metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    rate = float(y.mean())
    null = rate * (1.0 - rate)
    brier = float(np.mean((prediction - y) ** 2))
    return {
        "n": int(len(y)),
        "brier": brier,
        "bss": 100000.0 * (1.0 - brier / null),
        "target_mean": rate,
        "prediction_mean": float(prediction.mean()),
    }


def main() -> None:
    frame = load_labeled()
    season = frame["season"].to_numpy()
    label_ok = frame["label_ok"].to_numpy() == 1
    target = frame[SUCCESS].to_numpy(np.float64)
    report = {
        "method": "1 - (middle + reverse - mr + outside)",
        "note": (
            "Metrics use rows with reconstructable component labels. Centered BSS is "
            "diagnostic only and is never applied to saved predictions."),
        "folds": {},
    }
    for fold in (2023, 2024):
        mask = (season == fold) & label_ok
        component = {}
        sources = {}
        for name in ("middle", "reverse", "outside", "mr"):
            values, source = load_pred(name, fold)
            if values is None:
                raise FileNotFoundError(f"missing {name} fold {fold} prediction")
            component[name] = values
            sources[name] = source
        prediction = np.clip(
            1.0 - (component["middle"] + component["reverse"]
                   - component["mr"] + component["outside"]),
            1e-7, 1.0 - 1e-7)
        y = target[mask]
        if len(prediction) != len(y):
            raise RuntimeError(f"fold {fold} length mismatch")
        fold_frame = frame.loc[mask].reset_index(drop=True)
        metrics = bss(y, prediction)
        by_game_type = {}
        for game_type in ("R", "F"):
            sub = fold_frame["game_type"].astype(str).to_numpy() == game_type
            by_game_type[game_type] = subset_metrics(y[sub], prediction[sub])
        by_month = {}
        month_values = fold_frame["game_month"].to_numpy()
        for month in sorted(np.unique(month_values).tolist()):
            sub = month_values == month
            by_month[str(int(month))] = {
                "n": int(sub.sum()),
                "brier": float(np.mean((prediction[sub] - y[sub]) ** 2)),
            }
        report["folds"][str(fold)] = {
            "sources": sources,
            "metrics": metrics,
            "by_game_type": by_game_type,
            "by_month": by_month,
        }
    path = OUT / "final_validation_report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()

"""FA10C의 정직한 2022·2023 raw walk-forward 예측을 생성한다.

등록된 yn_fa10c_2024.csv와 같은 결합 전 raw 스케일을 사용한다:
0.10*LGB20 + 0.90*teamCB20. 평가 시즌 라벨은 학습, 조기 종료, calibration,
하이퍼파라미터 선택에 사용하지 않는다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as fa


def bss(target: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    rate = float(target.mean())
    brier = float(np.mean((pred - target) ** 2))
    score = 100000.0 * (1.0 - brier / (rate * (1.0 - rate)))
    return brier, score


def validate_output(rows: pd.DataFrame, pred: np.ndarray) -> None:
    if len(rows) != len(pred):
        raise ValueError("행 수 불일치")
    if rows[fa.ID_COL].isna().any() or rows[fa.ID_COL].duplicated().any():
        raise ValueError("row_id 결측 또는 중복")
    if not np.isfinite(pred).all():
        raise ValueError("예측에 NaN/Inf 존재")
    if np.any((pred < 0.0) | (pred > 1.0)):
        raise ValueError("예측 범위가 [0,1] 밖")
    if not 0.35 <= float(pred.mean()) <= 0.65:
        raise ValueError(f"예측 평균 안전 범위 밖: {pred.mean():.6f}")


def run(args: argparse.Namespace) -> None:
    data_dir = fa.find_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    raw_features = fa.raw_feature_list(data_dir)
    full = fa.load_train(data_dir, raw_features)
    metrics = {}
    for season in args.seasons:
        cutoff = season - 1
        fa.log(f"val{season}: season<={cutoff}만 학습")
        parts = fa.train_predictions_for_cutoff(
            full=full,
            raw_features=raw_features,
            cutoff=cutoff,
            valid_season=season,
            checkpoint_dir=checkpoint_dir,
            seeds=fa.LGB_SEEDS,
            include_numeric_cb=False,
        )
        pred = fa.combine_raw(parts)
        rows = full.loc[full["season"] == season].reset_index(drop=True)
        validate_output(rows, pred)
        output_path = output_dir / f"yn_fa10c_{season}.csv"
        pd.DataFrame(
            {
                fa.ID_COL: rows[fa.ID_COL].astype(str),
                "pred": pred,
            }
        ).to_csv(output_path, index=False)

        y = rows[fa.TARGET].to_numpy(dtype=float)
        season_metrics = {}
        for name, mask in {
            "all": np.ones(len(rows), dtype=bool),
            "R": rows["game_type"].to_numpy() == "R",
            "F": rows["game_type"].to_numpy() == "F",
        }.items():
            fold_brier, fold_bss = bss(y[mask], pred[mask])
            season_metrics[name] = {
                "n": int(mask.sum()),
                "brier": fold_brier,
                "bss": fold_bss,
                "true_mean": float(y[mask].mean()),
                "pred_mean": float(pred[mask].mean()),
            }
        metrics[str(season)] = season_metrics
        fa.log(
            f"저장 {output_path}: n={len(rows):,}, mean={pred.mean():.6f}, "
            f"all Brier={season_metrics['all']['brier']:.6f}, "
            f"R BSS={season_metrics['R']['bss']:.2f}"
        )

    report_path = checkpoint_dir / "val_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "prediction_contract": "raw=0.10*LGB20+0.90*teamCB20; no isotonic",
                "leakage_contract": "season S prediction trains only on season<=S-1",
                "metrics": metrics,
                "environment": fa.environment_versions(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    fa.log(f"검증 리포트: {report_path}")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023])
    parser.add_argument(
        "--output-dir",
        default=str(base.parents[2] / "val"),
        help="performance_tracking/val 경로",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(base.parent / "build" / "val_checkpoints"),
    )
    args = parser.parse_args()
    unsupported = sorted(set(args.seasons) - {2022, 2023, 2024})
    if unsupported:
        parser.error(f"지원하지 않는 시즌: {unsupported}")
    return args


if __name__ == "__main__":
    run(parse_args())

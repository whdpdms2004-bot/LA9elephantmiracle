"""D0와 동일 XGB 기준선의 예측 차이만 production base에 이식한다.

두 모델의 공통 계절 편향을 제거하고, 행 파생 피처가 바꾼 예측 residual만 평가한다.
스케일은 Val2023/Val2024에 동일하게 적용하며 test 예측 분포는 사용하지 않는다.
"""
from __future__ import annotations

import json
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SJ = HERE.parent
SRC = SJ / "claude" / "src"
sys.path.insert(0, str(SRC))
from harness import CACHE, TARGET, load, metrics

MO = SJ / "experiment" / "model_optimization"
OOF_DIR = MO / "enhanced_seed_oof_parts"
PROD = MO / "pitcher_cluster_matchup" / "reports" / "reverse20_submission_oof.parquet"
PRED = HERE / "outputs" / "single_xgb"
OUT = HERE / "outputs" / "d0_residual"
PREFIX = "confirm_xgboost_v2r200_tm500_robust_cpu_efull"
EPS = 1e-7


def load_base(df: pd.DataFrame, fold: int) -> np.ndarray:
    ids = df.loc[df["season"].eq(fold), "row_id"].to_numpy()
    if fold == 2024:
        prod = pd.read_parquet(PROD).set_index("row_id").reindex(ids)
        return prod["submit021_reverse20_s040_tabm"].to_numpy(np.float64)
    models = sorted({
        re.match(r"(.+)_fold\d{4}\.parquet", path.name).group(1)
        for path in OOF_DIR.glob("*.parquet")
    })
    values = []
    for name in models:
        path = OOF_DIR / f"{name}_fold{fold}.parquet"
        if path.exists():
            values.append(pd.read_parquet(path).set_index("row_id")
                          .reindex(ids)["prediction"].to_numpy(np.float64))
    if not values:
        raise FileNotFoundError(f"base predictions missing for fold {fold}")
    return np.mean(values, axis=0)


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, EPS, 1 - EPS)
    return np.log(values / (1 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=["D0", "C0", "C1", "C2"],
                        default="D0")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load()
    rows = []
    scales = [0.25, 0.50, 0.75, 1.00]
    for fold in (2023, 2024):
        valid = df["season"].eq(fold).to_numpy()
        y = df.loc[valid, TARGET].to_numpy(np.float64)
        game_type = df.loc[valid, "game_type"].astype(str).to_numpy()
        month = df.loc[valid, "game_month"].to_numpy()
        base = np.clip(load_base(df, fold), EPS, 1 - EPS)
        b0 = np.load(PRED / f"{PREFIX}_B0_{fold}.npy")
        candidate = np.load(PRED / f"{PREFIX}_{args.candidate}_{fold}.npy")
        if not (len(y) == len(base) == len(b0) == len(candidate)):
            raise AssertionError(f"row mismatch fold {fold}")
        ref = metrics(y, base)["bss_raw"]
        for space in ("prob", "logit"):
            direction = (candidate - b0 if space == "prob" else
                         logit(candidate) - logit(b0))
            for scale in scales:
                pred = (np.clip(base + scale * direction, EPS, 1 - EPS)
                        if space == "prob" else
                        sigmoid(logit(base) + scale * direction))
                score = metrics(y, pred)
                item = {
                    "fold": fold,
                    "space": space,
                    "scale": scale,
                    "bss": score["bss_raw"],
                    "dbss": score["bss_raw"] - ref,
                    "brier": score["brier"],
                    "pred_mean": float(pred.mean()),
                    "r_brier": float(np.mean((pred[game_type == "R"] - y[game_type == "R"]) ** 2)),
                    "f_brier": float(np.mean((pred[game_type == "F"] - y[game_type == "F"]) ** 2)),
                }
                for value in sorted(np.unique(month)):
                    mask = month == value
                    item[f"month_{int(value):02d}_brier"] = float(
                        np.mean((pred[mask] - y[mask]) ** 2))
                rows.append(item)
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.candidate.lower()
    result.to_csv(OUT / f"{stem}_residual_grid.csv", index=False)
    pivot = result.pivot_table(index=["space", "scale"], columns="fold", values="dbss")
    pivot["worst"] = pivot.min(axis=1)
    pivot["sum"] = pivot[[2023, 2024]].sum(axis=1)
    pivot = pivot.sort_values(["worst", "sum"], ascending=False)
    print(pivot.round(3).to_string())
    best = pivot.reset_index().iloc[0].to_dict()
    (OUT / f"{stem}_summary.json").write_text(
        json.dumps({"candidate": args.candidate,
                    "selection": "maximize worst-fold Delta BSS", "best": best},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

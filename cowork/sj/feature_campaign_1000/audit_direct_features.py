"""원본 입력 피처의 스키마와 시간 안전 단변량 Brier 기여를 감사한다.

지정한 검증 시즌에서 각 열의 효과를 다음처럼 잰다.
  1. 검증 시즌 이전에서만 수치형 bin edge 또는 범주 lookup을 fit
  2. 각 그룹 Target 평균을 K=200으로 전체 평균에 수축
  3. 학습 Target으로 외삽한 2024 base rate에 그룹 residual만 더함
  4. 2024 Brier/BSS 및 상수 base 대비 Delta BSS 기록

이는 최종 모델이 아니라 피처/상호작용 우선순위용 진단이다.
"""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC = Path(__file__).resolve().parents[1] / "claude" / "src"
sys.path.insert(0, str(SRC))
from harness import TARGET, forecast_base_rate, load, metrics

OUT = Path(__file__).resolve().parent / "outputs" / "audit"
OUT.mkdir(parents=True, exist_ok=True)
K = 200.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid", type=int, choices=[2023, 2024], default=2024)
    return parser.parse_args()


def family(column: str) -> str:
    if column in {"season", "game_month", "game_dayofweek"}:
        return "time"
    if column in {"inning", "top_bottom", "game_type", "balls_before",
                  "strikes_before", "outs_before"}:
        return "game_state"
    if column.startswith("run_") or column.startswith("score_diff"):
        return "score"
    if column.startswith("runner_") or column in {"num_runners_on", "base_state"}:
        return "runners"
    if column in {"home_win_expectancy", "away_win_expectancy", "li"}:
        return "leverage"
    if column in {"pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
                  "pitcher_team_id", "batter_team_id"}:
        return "identity"
    if column.startswith("asof_pitcher_prev"):
        return "recent_form"
    if column.startswith("asof_batter"):
        return "batter_history"
    if "pitchmix" in column or column.endswith(("fastball_rate", "breaking_rate",
                                                  "offspeed_rate")):
        return "pitch_mix"
    if column.startswith("asof_pitcher"):
        return "pitcher_history"
    return "other"


def numeric_groups(train: pd.Series, valid: pd.Series, bins: int = 30):
    finite = pd.to_numeric(train, errors="coerce").dropna()
    if finite.nunique() <= bins:
        return train.astype(str).fillna("__NA__"), valid.astype(str).fillna("__NA__")
    edges = np.unique(np.nanquantile(finite.to_numpy(np.float64),
                                     np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return train.astype(str).fillna("__NA__"), valid.astype(str).fillna("__NA__")
    edges[0], edges[-1] = -np.inf, np.inf
    return (pd.cut(pd.to_numeric(train, errors="coerce"), edges, labels=False,
                   include_lowest=True).fillna(-1).astype("int16"),
            pd.cut(pd.to_numeric(valid, errors="coerce"), edges, labels=False,
                   include_lowest=True).fillna(-1).astype("int16"))


def evaluate_column(train: pd.DataFrame, valid: pd.DataFrame, column: str,
                    base_rate: float, base_bss: float):
    tr_col, va_col = train[column], valid[column]
    if pd.api.types.is_numeric_dtype(tr_col):
        gtr, gva = numeric_groups(tr_col, va_col)
    else:
        gtr = tr_col.astype(str).fillna("__NA__")
        gva = va_col.astype(str).fillna("__NA__")
    grouped = (pd.DataFrame({"g": gtr.to_numpy(), "y": train[TARGET].to_numpy()})
               .groupby("g", dropna=False)["y"].agg(["sum", "size"]))
    global_train = float(train[TARGET].mean())
    smoothed = (grouped["sum"] + K * global_train) / (grouped["size"] + K)
    residual = smoothed - global_train
    pred = np.clip(base_rate + pd.Series(gva).map(residual).fillna(0.0).to_numpy(),
                   1e-6, 1 - 1e-6)
    score = metrics(valid[TARGET].to_numpy(), pred)
    return score["bss_raw"], score["bss_raw"] - base_bss, float(pred.mean())


def main():
    valid_season = parse_args().valid
    df = load()
    train = df[df["season"] < valid_season]
    valid = df[df["season"] == valid_season]
    tr_mask = df["season"].to_numpy() < valid_season
    base_rate = forecast_base_rate(df, tr_mask, valid_season)
    base_pred = np.full(len(valid), base_rate)
    base_score = metrics(valid[TARGET].to_numpy(), base_pred)
    rows = []
    for column in [c for c in df.columns if c not in ("row_id", TARGET)]:
        bss, delta, pred_mean = evaluate_column(
            train, valid, column, base_rate, base_score["bss_raw"])
        numeric = pd.to_numeric(df[column], errors="coerce")
        rows.append({
            "column": column,
            "family": family(column),
            "dtype": str(df[column].dtype),
            "n_unique": int(df[column].nunique(dropna=True)),
            "missing_rate": float(df[column].isna().mean()),
            "min": float(numeric.min()) if numeric.notna().any() else np.nan,
            "median": float(numeric.median()) if numeric.notna().any() else np.nan,
            "max": float(numeric.max()) if numeric.notna().any() else np.nan,
            "univariate_bss": bss,
            "delta_vs_constant": delta,
            "pred_mean": pred_mean,
        })
        print(f"{column:<45} {delta:+9.3f}", flush=True)
    result = pd.DataFrame(rows).sort_values(
        "delta_vs_constant", ascending=False)
    result.to_csv(OUT / f"direct_feature_audit_{valid_season}.csv", index=False)
    seasonal = df.groupby("season")[TARGET].agg(["size", "mean"]).reset_index()
    seasonal.to_csv(OUT / "season_target_rate.csv", index=False)
    summary = {
        "train_shape": list(df.shape),
        "input_columns_excluding_row_id": int(len(rows)),
        "valid_season": valid_season,
        "valid_rows": int(len(valid)),
        "constant_base_rate": base_rate,
        "constant_base_brier": base_score["brier"],
        "constant_base_bss": base_score["bss_raw"],
        "top_10": result[["column", "family", "delta_vs_constant"]]
        .head(10).to_dict("records"),
    }
    (OUT / f"summary_{valid_season}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

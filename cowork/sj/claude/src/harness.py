"""공통 실험 하네스.

모든 Phase B~D 실험이 이 모듈을 통해 돌아간다. 목적은 두 가지다.
  1. 같은 fold / 같은 지표 / 같은 출력 스키마로 비교 가능성을 보장한다.
  2. train.csv 로딩(60~90s)을 parquet 캐시(3~5s)로 대체해 실험 회전을 올린다.

사용:
    from harness import load, add_stateless, folds, fit_predict, metrics, log_result
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]              # cowork/sj/claude
DATA_CSV = ROOT.parent / "data" / "train.csv"
CACHE = ROOT / "cache"
OUT = ROOT / "outputs"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

TARGET = "control_success"
PARQUET = CACHE / "train.parquet"

INT_COLS = ["season", "game_month", "game_dayofweek", "inning",
            "balls_before", "strikes_before", "outs_before",
            "run_top_before", "run_bot_before", "run_total_before",
            "score_diff_home", "score_diff_pitcher_team",
            "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
            "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
            "pitcher_team_id", "batter_team_id"]
STR_COLS = ["top_bottom", "game_type", "base_state"]

# 채택된 GBDT 기본 파라미터. 팀 검증에서 얕은 트리(18~24 leaves)가 안정적이었다.
BASE_PARAMS = dict(max_depth=0, grow_policy="lossguide", max_leaves=18, eta=0.03,
                   subsample=0.8, colsample_bytree=0.6, min_child_weight=64,
                   reg_lambda=2.0, objective="binary:logistic",
                   eval_metric="logloss", tree_method="hist", max_bin=256,
                   device="cuda")


def build_cache(force: bool = False) -> Path:
    """train.csv -> parquet. dtype을 줄여 로딩과 메모리를 함께 낮춘다."""
    if PARQUET.exists() and not force:
        return PARQUET
    t0 = time.perf_counter()
    df = pd.read_csv(DATA_CSV)
    for c in INT_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes("float64"):
        df[c] = df[c].astype("float32")
    for c in STR_COLS:
        df[c] = df[c].astype("category")
    df[TARGET] = df[TARGET].astype("int8")
    df.to_parquet(PARQUET, index=False)
    print(f"cache built {PARQUET} in {time.perf_counter()-t0:.1f}s  "
          f"{df.shape}  {df.memory_usage(deep=True).sum()/2**20:.0f}MB", flush=True)
    return PARQUET


def load() -> pd.DataFrame:
    build_cache()
    t0 = time.perf_counter()
    df = pd.read_parquet(PARQUET)
    print(f"loaded {df.shape} in {time.perf_counter()-t0:.1f}s", flush=True)
    return df


RATE_SPECS = [
    ("pitcher_success", "asof_pitcher_success_rate", "asof_pitcher_n"),
    ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("batter_success", "asof_batter_success_rate", "asof_batter_n"),
    ("batter_middle", "asof_batter_middle_rate", "asof_batter_n"),
]


def make_priors(fit_df: pd.DataFrame) -> dict:
    """prior는 반드시 학습 fold에서만 계산한다."""
    return {
        "pitcher_success": float(fit_df[TARGET].mean()),
        "pitcher_reverse": float(fit_df["asof_pitcher_reverse_rate"].median()),
        "pitcher_middle": float(fit_df["asof_pitcher_middle_rate"].median()),
        "batter_success": float(fit_df[TARGET].mean()),
        "batter_middle": float(fit_df["asof_batter_middle_rate"].median()),
    }


def add_stateless(df: pd.DataFrame, priors: dict, strength: float = 200.0) -> pd.DataFrame:
    """README §10.7 파생. Target을 쓰지 않고 행 단위로만 계산한다."""
    out = df.copy()
    out["count_state"] = (out["balls_before"].astype(int) * 3
                          + out["strikes_before"].astype(int)).astype("int16")
    out["handedness_matchup"] = (out["pitcher_hand"].astype(int) * 2
                                 + out["batter_hand"].astype(int)).astype("int16")
    out["runner_out_state"] = (out["num_runners_on"].astype(int) * 3
                               + out["outs_before"].astype(int)).astype("int16")
    out["score_abs"] = out["score_diff_pitcher_team"].abs().astype("int16")
    out["late_inning"] = (out["inning"] >= 7).astype("int8")
    out["high_leverage"] = (out["li"] >= 2).astype("int8")
    out["log1p_asof_pitcher_n"] = np.log1p(out["asof_pitcher_n"]).astype("float32")
    out["log1p_asof_batter_n"] = np.log1p(out["asof_batter_n"]).astype("float32")
    for k in (1, 3, 5):
        out[f"pitcher_success_delta_prev{k}"] = (
            out[f"asof_pitcher_prev{k}_game_success_rate"]
            - out["asof_pitcher_success_rate"]).astype("float32")
        out[f"pitcher_middle_delta_prev{k}"] = (
            out[f"asof_pitcher_prev{k}_game_middle_rate"]
            - out["asof_pitcher_middle_rate"]).astype("float32")
    out["ball_strike_gap"] = (out["asof_pitcher_ball_rate"]
                              - out["asof_pitcher_strike_rate"]).astype("float32")

    for name, rate_col, n_col in RATE_SPECS:
        n = out[n_col].astype("float32")
        rate = out[rate_col].fillna(priors[name]).astype("float32")
        out[f"{name}_is_missing"] = out[rate_col].isna().astype("int8")
        out[f"{name}_smoothed"] = ((n * rate + strength * priors[name])
                                   / (n + strength)).astype("float32")
        out[f"{name}_reliability"] = (n / (n + strength)).astype("float32")
    return out


def encode(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in STR_COLS:
        if c in out and str(out[c].dtype) == "category":
            out[c] = out[c].cat.codes.astype("int8")
    return out


def folds(df: pd.DataFrame, valid_seasons=(2022, 2023, 2024)) -> dict:
    """순방향 split. 학습은 검증 시즌 이전만."""
    s = df["season"].to_numpy()
    return {v: (s < v, s == v) for v in valid_seasons}


def forecast_base_rate(df: pd.DataFrame, tr_mask, valid_season: float) -> float:
    """검증 시즌의 성공률을 학습 시즌 라벨만으로 예측한다 (random walk + drift).

    규칙: 마지막 학습 시즌 성공률 + (학습 구간 연평균 변화량) x (연도 간격).
    모든 fold에 동일하게 적용하는 사전 등록 규칙이다. 검증 시즌 라벨을 쓰지 않는다.

    시즌 drift가 -7.86%p/6년으로 크기 때문에 이 보정 없이는 GBDT의 base_score가
    학습 구간 pooled 평균에 고정되어 검증 시즌에서 체계적 편향이 생긴다.
    """
    sub = df.loc[tr_mask]
    rates = sub.groupby("season")[TARGET].mean().sort_index()
    seasons = rates.index.to_numpy(dtype=float)
    last_s, last_r = float(seasons[-1]), float(rates.iloc[-1])
    if len(rates) < 2:
        return last_r
    per_year = (last_r - float(rates.iloc[0])) / (last_s - float(seasons[0]))
    return float(np.clip(last_r + per_year * (valid_season - last_s), 0.01, 0.99))


def metrics(y, p, game_type=None, month=None) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    null = y.mean() * (1 - y.mean())
    brier = float(np.mean((p - y) ** 2))
    res = {"n": int(len(y)), "target_mean": float(y.mean()),
           "pred_mean": float(p.mean()), "brier": brier,
           "normalized_brier": brier / null,
           "bss_raw": float(100000 * (1 - brier / null))}
    if game_type is not None:
        gt = np.asarray(game_type)
        for tag in ("R", "F"):
            m = gt == tag
            if m.sum():
                yn, pn = y[m], p[m]
                nb = yn.mean() * (1 - yn.mean())
                b = float(np.mean((pn - yn) ** 2))
                res[f"{tag.lower()}_bss"] = float(100000 * (1 - b / nb))
                res[f"{tag.lower()}_n"] = int(m.sum())
    if month is not None:
        mo = np.asarray(month)
        res["month_brier"] = json.dumps(
            {int(k): round(float(np.mean((p[mo == k] - y[mo == k]) ** 2)), 8)
             for k in sorted(set(mo.tolist()))})
    return res


def fit_predict(X_tr, y_tr, X_va, y_va, params=None, seed=0,
                num_boost_round=2000, early_stopping_rounds=100, verbose=False):
    prm = {**BASE_PARAMS, **(params or {}), "seed": seed}
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    t0 = time.perf_counter()
    bst = xgb.train(prm, dtr, num_boost_round=num_boost_round,
                    evals=[(dva, "val")],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=500 if verbose else False)
    p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    return p, {"best_iter": int(bst.best_iteration),
               "elapsed_sec": round(time.perf_counter() - t0, 2)}, bst


def log_result(row: dict, name: str):
    """실험 기록을 append. 기존 줄은 고치지 않는다."""
    path = OUT / f"{name}.csv"
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path

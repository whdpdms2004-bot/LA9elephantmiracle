# -*- coding: utf-8 -*-
"""cowork/ye/model_v3.ipynb 의 M2_contam + WEIGHTED_DIRECT(f_recent_strong) 구조로
performance_tracking 규격 OOF 예측 3개(val2022/2023/2024)를 만든다.

## 원본(model_v3.ipynb)과 같게 한 것
- build_submitted_fold: 공통 전처리 + reliability + leakage-safe platoon EB
  + game_type + contamination 3피처 (as-of, 시즌 경계 지킴 — 검증 완료. 이름은
  "contamination" 이지만 과거 시즌 이력만 쓰는 정상적인 as-of 피처다)
- WEIGHT_RECIPES / make_sample_weight: f_recent_strong (노트북 자체 판정 승자,
  2024 screening +28.0, 2023 stress에서도 방향 일치)
- CAT_PARAMS: Logloss/BrierScore, depth=6, l2_leaf_reg=10.0, lr=0.03, border_count=128

## 원본과 다르게 한 것 — 반드시 필요한 수정
노트북의 fit_weighted_direct() 는 eval_set=(va_df,...) + early_stopping_rounds=200 +
use_best_model=True 를 쓴다 — 평가 시즌 자신의 라벨을 조기 종료에 쓰는 것으로,
hw 가 이미 걸렸던 오염 패턴과 동일하다 (팀 규칙 2 위반).
여기서는 eval_set 없이 **고정 iteration=349** 로 학습한다. 349 는 노트북 자신의
5-seed 실험(run_direct_5seed)이 계산해 둔 median_iteration 값이자, 노트북의
(미실행) 최종 10-seed 빌드 셀이 쓰려고 했던 바로 그 값이다 — 임의로 새로 고른 게
아니라 노트북이 이미 "최종 구조"로 지목한 고정 하이퍼파라미터를 그대로 세 폴드에
동일하게 적용한다 (hw 에게 준 지침과 동일한 원칙: 폴드마다 하이퍼는 고정).

## 알려진 문제
val2023 은 F 부분군에서 붕괴한다 (단독 all BSS -1,230.3, F -16,825 — 2022->2023
game_type 구조단절 때문으로 보임). R 만 보면 양호(547.9). 팀의 다른 모델(hw/yn)도
2023 에서 동일하게 무너지는 걸 확인했다 — ye_hand 만의 결함이 아니라 이 데이터셋
자체의 특성으로 보인다 (§6. 남은 것 참고).

## 산출물
performance_tracking/val/ye_hand_2022.csv  (<=2021 학습)
performance_tracking/val/ye_hand_2023.csv  (<=2022 학습)
performance_tracking/val/ye_hand_2024.csv  (<=2023 학습)

## 실행
cd performance_tracking/models/ye_hand/ && python train_ye_hand.py
(저장소 루트의 data/train.csv 를 읽는다. 로컬에 없으면 대회 페이지에서 받아
data/train.csv 로 배치할 것)
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(m):
    print(m, flush=True)


REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data"
OUT_DIR = REPO / "performance_tracking" / "val"

TARGET = "control_success"
ID_COL = "row_id"

SEEDS = [11, 22, 33]
FIXED_ITERATIONS = 349  # model_v3.ipynb run_direct_5seed()['median_iteration']
RECIPE_NAME = "f_recent_strong"

CAT_PARAMS = dict(
    loss_function="Logloss",
    eval_metric="BrierScore",
    iterations=FIXED_ITERATIONS,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10.0,
    random_strength=0.5,
    border_count=128,
    verbose=False,
    allow_writing_files=False,
)

RELIABILITY_K = 200.0
PLATOON_K = 200.0

CATEGORICAL_CANDIDATES = [
    "game_dayofweek", "top_bottom", "game_type", "base_state",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
]

CONTAM_FEATURES = [
    "fe_pitcher_futures_share",
    "fe_batter_futures_share",
    "fe_pitcher_prior_n_log",
]

BASE_DROP_MODEL = {
    TARGET, ID_COL, "pitcher_id", "batter_id",
    "pitcher_eb", "pitcher_handmatch_eb", "p_n", "ph_n",
}

WEIGHT_RECIPES = {
    "equal": {"kind": "equal"},
    "global_recent": {
        "kind": "global_recent",
        "age_weight": {0: 1.00, 1: 0.85, 2: 0.70, 3: 0.60},
        "older": 0.50,
    },
    "f_recent_moderate": {
        "kind": "f_recent",
        "age_weight": {0: 1.00, 1: 0.80, 2: 0.60, 3: 0.50},
        "older": 0.50,
    },
    "f_recent_strong": {
        "kind": "f_recent",
        "age_weight": {0: 1.00, 1: 0.60, 2: 0.40, 3: 0.30},
        "older": 0.30,
    },
    "f_recent_low_n": {
        "kind": "f_recent_low_n",
        "age_weight": {0: 1.00, 1: 0.80, 2: 0.60, 3: 0.50},
        "older": 0.50,
        "low_n_floor": 0.60,
        "low_n_k": 200.0,
    },
}


def make_sample_weight(tr_raw, recipe_name):
    recipe = WEIGHT_RECIPES[recipe_name]
    w = pd.Series(np.ones(len(tr_raw)), index=tr_raw.index, dtype=float)
    if recipe["kind"] == "equal":
        return w
    latest = int(tr_raw["season"].max())
    age = latest - tr_raw["season"].astype(int)
    recency = age.map(lambda a: recipe["age_weight"].get(int(a), recipe["older"])).astype(float)
    if recipe["kind"] == "global_recent":
        w *= recency
    else:
        is_f = tr_raw["game_type"].astype(str).eq("F")
        w.loc[is_f] *= recency.loc[is_f]
        if recipe["kind"] == "f_recent_low_n":
            if "asof_pitcher_n" in tr_raw.columns:
                n = pd.to_numeric(tr_raw["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0)
            else:
                n = pd.Series(0.0, index=tr_raw.index)
            k = recipe["low_n_k"]
            floor = recipe["low_n_floor"]
            rel = n / (n + k)
            w *= floor + (1 - floor) * rel
    return w.clip(0.05, 1.0)


def fit_preprocess_state(df_fit):
    state = {"target_prior": float(df_fit[TARGET].mean()), "rate_median": {}}
    rate_cols = [c for c in df_fit.columns if c.startswith("asof_") and c.endswith("_rate")]
    for c in rate_cols:
        med = df_fit[c].median()
        if pd.isna(med):
            med = state["target_prior"] if "success_rate" in c else 0.0
        state["rate_median"][c] = float(med)
    return state


def apply_common_preprocess(df, state):
    out = df.copy()
    for c in CATEGORICAL_CANDIDATES:
        if c in out.columns:
            out[c] = out[c].astype("object").where(out[c].notna(), "__MISSING__").astype(str)
    important_missing_cols = [
        "asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate", "asof_batter_success_rate",
    ]
    for c in important_missing_cols:
        if c in out.columns:
            out[f"{c}__missing"] = out[c].isna().astype("int8")
    career = "asof_pitcher_success_rate"
    if career in out.columns:
        fallback = state["rate_median"].get(career, state["target_prior"])
        out[career] = out[career].fillna(fallback)
        for c in ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
                  "asof_pitcher_prev5_game_success_rate"]:
            if c in out.columns:
                out[c] = out[c].fillna(out[career])
    if "asof_batter_success_rate" in out.columns:
        out["asof_batter_success_rate"] = out["asof_batter_success_rate"].fillna(
            state["rate_median"].get("asof_batter_success_rate", state["target_prior"]))
    rate_cols = [c for c in df.columns if c.startswith("asof_") and c.endswith("_rate")]
    for c in rate_cols:
        if c in out.columns:
            out[c] = out[c].fillna(state["rate_median"].get(c, 0.0))
    return out


def add_reliability_features(df):
    out = df.copy()
    for c in ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]:
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(lower=0)
            out[f"log1p__{c}"] = np.log1p(x)
            out[f"reliability__{c}"] = x / (x + RELIABILITY_K)
    return out


def fit_platoon_lookup(df_fit, k=PLATOON_K):
    prior = float(df_fit[TARGET].mean())
    gp = (df_fit.groupby("pitcher_id", dropna=False)[TARGET].agg(["sum", "count"]).reset_index()
          .rename(columns={"sum": "p_success", "count": "p_n"}))
    gp["pitcher_eb"] = (gp["p_success"] + k * prior) / (gp["p_n"] + k)
    gh = (df_fit.groupby(["pitcher_id", "batter_hand"], dropna=False)[TARGET].agg(["sum", "count"]).reset_index()
          .rename(columns={"sum": "ph_success", "count": "ph_n"}))
    gh["pitcher_handmatch_eb"] = (gh["ph_success"] + k * prior) / (gh["ph_n"] + k)
    return {
        "prior": prior,
        "pitcher": gp[["pitcher_id", "p_n", "pitcher_eb"]],
        "hand": gh[["pitcher_id", "batter_hand", "ph_n", "pitcher_handmatch_eb"]],
    }


def apply_platoon_lookup(df, lookup, k=PLATOON_K):
    out = df.copy()
    out = out.merge(lookup["pitcher"], on="pitcher_id", how="left")
    out = out.merge(lookup["hand"], on=["pitcher_id", "batter_hand"], how="left")
    prior = lookup["prior"]
    out["p_n"] = out["p_n"].fillna(0)
    out["ph_n"] = out["ph_n"].fillna(0)
    out["pitcher_eb"] = out["pitcher_eb"].fillna(prior)
    out["pitcher_handmatch_eb"] = out["pitcher_handmatch_eb"].fillna(out["pitcher_eb"])
    out["platoon_split_eb"] = out["pitcher_handmatch_eb"] - out["pitcher_eb"]
    out["platoon_n_reliability"] = out["ph_n"] / (out["ph_n"] + k)
    return out


def add_temporal_platoon_features(df_train, k=PLATOON_K):
    out = df_train.copy()
    generated = pd.DataFrame(index=out.index, columns=["platoon_split_eb", "platoon_n_reliability"], dtype=float)
    for season in sorted(out["season"].dropna().unique()):
        idx = out.index[out["season"] == season]
        history = out.loc[out["season"] < season]
        if len(history) == 0:
            generated.loc[idx, :] = 0.0
            continue
        enc = apply_platoon_lookup(out.loc[idx], fit_platoon_lookup(history, k=k), k=k)
        generated.loc[idx, "platoon_split_eb"] = enc["platoon_split_eb"].values
        generated.loc[idx, "platoon_n_reliability"] = enc["platoon_n_reliability"].values
    out["platoon_split_eb"] = generated["platoon_split_eb"].fillna(0.0)
    out["platoon_n_reliability"] = generated["platoon_n_reliability"].fillna(0.0)
    return out


def fit_futures_lookup(df_history):
    hist = df_history.copy()
    is_f = hist["game_type"].astype(str).eq("F").astype(float)
    tmp = pd.DataFrame({
        "pitcher_id": hist["pitcher_id"].values,
        "batter_id": hist["batter_id"].values,
        "_is_f": is_f.values,
    })
    p = tmp.groupby("pitcher_id", dropna=False)["_is_f"].agg(["mean", "count"])
    b = tmp.groupby("batter_id", dropna=False)["_is_f"].mean()
    return {
        "pitcher_futures_share": p["mean"].to_dict(),
        "pitcher_prior_n": p["count"].to_dict(),
        "batter_futures_share": b.to_dict(),
    }


def apply_futures_lookup(df, lookup):
    out = df.copy()
    out["fe_pitcher_futures_share"] = out["pitcher_id"].map(lookup["pitcher_futures_share"]).astype(float)
    out["fe_batter_futures_share"] = out["batter_id"].map(lookup["batter_futures_share"]).astype(float)
    pn = out["pitcher_id"].map(lookup["pitcher_prior_n"]).astype(float)
    out["fe_pitcher_prior_n_log"] = np.log1p(pn)
    return out


def add_temporal_futures_features(df_train_raw):
    out = df_train_raw.copy()
    generated = pd.DataFrame(index=out.index, columns=CONTAM_FEATURES, dtype=float)
    for season in sorted(out["season"].dropna().unique()):
        idx = out.index[out["season"] == season]
        history = out.loc[out["season"] < season]
        if len(history) == 0:
            continue
        enc = apply_futures_lookup(out.loc[idx], fit_futures_lookup(history))
        generated.loc[idx, CONTAM_FEATURES] = enc[CONTAM_FEATURES].values
    for c in CONTAM_FEATURES:
        out[c] = generated[c]
    return out


def get_model_features(df):
    return [c for c in df.columns if c not in BASE_DROP_MODEL]


def get_cat_feature_indices(features):
    cat_names = [c for c in CATEGORICAL_CANDIDATES if c in features]
    return [features.index(c) for c in cat_names], cat_names


def build_submitted_fold(full_train, val_year):
    """model_v3.ipynb cell11 그대로 — 최종 제출했던 M2_contam 구조."""
    tr_raw = full_train.loc[full_train["season"] < val_year].copy()
    va_raw = full_train.loc[full_train["season"] == val_year].copy()
    if len(tr_raw) == 0 or len(va_raw) == 0:
        raise ValueError(f"val_year={val_year}: rows 부족")

    state = fit_preprocess_state(tr_raw)
    tr = add_reliability_features(apply_common_preprocess(tr_raw, state))
    va = add_reliability_features(apply_common_preprocess(va_raw, state))

    tr = add_temporal_platoon_features(tr, k=PLATOON_K)
    va = apply_platoon_lookup(va, fit_platoon_lookup(tr, k=PLATOON_K), k=PLATOON_K)

    tr_contam = add_temporal_futures_features(tr_raw)
    for c in CONTAM_FEATURES:
        tr[c] = tr_contam[c].values

    va_contam = apply_futures_lookup(va_raw, fit_futures_lookup(tr_raw))
    for c in CONTAM_FEATURES:
        va[c] = va_contam[c].values

    return tr, va, tr_raw, va_raw


def bss(p, y):
    r = y.mean()
    return 100000.0 * max(0.0, (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r))))


def main():
    t0 = time.time()
    train_path = DATA_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} 가 없다. 대회 원본은 커밋하지 않으므로 로컬에 직접 배치해야 한다."
        )
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    train.columns = [c.strip("﻿") for c in train.columns]
    log(f"train {len(train):,}행 · {train.shape[1]}열  ({time.time()-t0:.0f}s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for val_year in (2022, 2023, 2024):
        tf = time.time()
        tr_df, va_df, tr_raw, va_raw = build_submitted_fold(train, val_year)
        features = get_model_features(tr_df)
        cat_idx, cat_names = get_cat_feature_indices(features)
        w = make_sample_weight(tr_raw, RECIPE_NAME)
        weight_map = pd.Series(w.values, index=tr_raw[ID_COL].values)
        sample_weight = tr_df[ID_COL].map(weight_map).fillna(1.0).values

        log(f"\nfold{val_year}  학습 {len(tr_df):,}행 · 검증 {len(va_df):,}행 · "
            f"피처 {len(features)}개 (범주 {len(cat_names)}개)  (전처리 {time.time()-tf:.0f}s)")

        acc = np.zeros(len(va_df))
        for seed in SEEDS:
            ts = time.time()
            params = dict(CAT_PARAMS)
            params["random_seed"] = seed
            model = CatBoostClassifier(**params)
            model.fit(
                tr_df[features], tr_df[TARGET].astype(int),
                cat_features=cat_idx, sample_weight=sample_weight,
                verbose=False,
            )
            p = model.predict_proba(va_df[features])[:, 1]
            acc += p
            log(f"    seed{seed}  {time.time()-ts:.0f}s  단독 BSS {bss(p, va_df[TARGET].to_numpy(float)):.1f}")
            del model
            gc.collect()

        pred = acc / len(SEEDS)
        y = va_df[TARGET].to_numpy(float)
        log(f"  {len(SEEDS)}시드 평균  BSS {bss(pred, y):.1f}  "
            f"(평균예측 {pred.mean():.4f} · 실제 {y.mean():.4f})")

        out = OUT_DIR / f"ye_hand_{val_year}.csv"
        pd.DataFrame({"row_id": va_df[ID_COL].to_numpy(), "pred": pred}).to_csv(out, index=False)
        log(f"  -> {out}")
        del tr_df, va_df, tr_raw, va_raw

    log(f"\n총 {(time.time()-t0)/60:.1f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

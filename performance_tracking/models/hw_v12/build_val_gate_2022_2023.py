"""hw_v12 관문시즌(2022·2023) 정직 예측 생성 -- TEAM_BRIEF.md §4 / AGENT_PROMPTS.md [hw] 대응.

★ 반드시 피할 것 (이미 한 번 밟은 함정, sj가 build_val2024_pred_v12.py에서 발견):
    - eval_set=(x_val, y_val) + use_best_model=True 로 "평가 시즌" 라벨을
      조기종료에 쓰면 안 된다 (val2024 808.9 -> 정직본 720.6로 하락한 원인).
    - league_avg를 raw 전체(미래 시즌 포함)로 구하면 안 된다.

이 스크립트의 해법 -- 내부 홀드아웃으로 조기종료:
    각 타깃시즌(2022/2023)에 대해 fit = season < 타깃시즌.
    best_iteration은 fit **내부**의 마지막 시즌만 떼어(internal monitor) 구하고,
    그 iteration 수만큼 fit 전체(내부 분할 없이)로 다시 학습해서 타깃시즌에 예측한다.
    타깃시즌 라벨은 학습 루프 어디에도 안 들어간다.

    예) val2022 (fit=2019~2021):
        내부: train=2019~2020, monitor=2021 -> best_iter 결정
        최종: fit=2019~2021 전체, iterations=best_iter(고정), eval_set 없음 -> 2022 예측
    예) val2023 (fit=2019~2022):
        내부: train=2019~2021, monitor=2022 -> best_iter 결정
        최종: fit=2019~2022 전체, iterations=best_iter(고정), eval_set 없음 -> 2023 예측

하이퍼파라미터·피처셋은 train_best_model_v12.py와 완전히 동일 (팀 규칙: 바뀌면
val2024와 비교가 안 됨). 4시드 배깅 (sj의 정직본 재생산과 동일 시드 수, 속도 고려).

출력 (규칙 1 개정 스펙, row_id,pred 2컬럼):
    performance_tracking/val/hw_v12_2022.csv
    performance_tracking/val/hw_v12_2023.csv

실행:
    python performance_tracking/models/hw_v12/build_val_gate_2022_2023.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parents[3]  # models/hw_v12 -> models -> performance_tracking -> 저장소 루트
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "performance_tracking" / "val"

ID = "row_id"
TARGET = "control_success"
BASELINE_CATS = ["top_bottom", "game_type", "base_state",
                 "pitcher_team_id", "batter_team_id", "count_state",
                 "pitcher_hand", "batter_hand", "handedness_matchup"]
PREV_PAIRS = (1, 3, 5)
K_PLATOON = 300.0
SEEDS = [2026, 2027, 2028, 2029]  # 4시드, sj 정직본 재생산과 동일 수

CB_PARAMS = dict(  # train_best_model_v12.py와 완전 동일 -- 바꾸지 않는다
    loss_function="Logloss", eval_metric="BrierScore", depth=6,
    learning_rate=0.03, l2_leaf_reg=25, random_strength=0.6, border_count=128,
    thread_count=-1, grow_policy="Depthwise", boosting_type="Plain",
    bootstrap_type="Bernoulli", subsample=0.7, rsm=0.7,
    verbose=False, od_type="Iter", od_wait=100, allow_writing_files=False,
)
MAX_ITER = 1500

TARGETS = [
    (2022, 2021),  # (val_season, internal_monitor_season)
    (2023, 2022),
]


def log(msg, t0):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def add_trend(df):
    x = df.copy()
    for k in PREV_PAIRS:
        recent = f"asof_pitcher_prev{k}_game_success_rate"
        x[f"trend_prev{k}"] = x[recent] - x["asof_pitcher_success_rate"]
        x[f"trend_abs_prev{k}"] = x[f"trend_prev{k}"].abs()
    return x


def add_count_state(df):
    x = df.copy()
    x["count_state"] = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    return x


def add_matchup(df):
    x = df.copy()
    x["handedness_matchup"] = x["pitcher_hand"].astype(str) + "_" + x["batter_hand"].astype(str)
    return x


def add_platoon_prior_cumulative(df, league_avg, K=K_PLATOON):
    x = df.copy()
    cum_success_ph = x.groupby(["pitcher_id", "batter_hand"])[TARGET].cumsum()
    cum_n_ph = x.groupby(["pitcher_id", "batter_hand"]).cumcount() + 1
    prior_success_ph = cum_success_ph - x[TARGET]
    prior_n_ph = cum_n_ph - 1
    eb_ph = (prior_success_ph + K * league_avg) / (prior_n_ph + K)
    cum_success_p = x.groupby("pitcher_id")[TARGET].cumsum()
    cum_n_p = x.groupby("pitcher_id").cumcount() + 1
    prior_success_p = cum_success_p - x[TARGET]
    prior_n_p = cum_n_p - 1
    eb_p = (prior_success_p + K * league_avg) / (prior_n_p + K)
    x["platoon_split"] = eb_ph - eb_p
    x["platoon_n"] = np.log1p(prior_n_ph)
    return x


def build_platoon_lookup(source_df, league_avg, K=K_PLATOON):
    ph = source_df.groupby(["pitcher_id", "batter_hand"])[TARGET].agg(n="count", s="sum").reset_index()
    p = source_df.groupby("pitcher_id")[TARGET].agg(p_n="count", p_s="sum").reset_index()
    ph = ph.merge(p, on="pitcher_id", how="left")
    ph["eb_ph"] = (ph["s"] + K * league_avg) / (ph["n"] + K)
    ph["eb_p"] = (ph["p_s"] + K * league_avg) / (ph["p_n"] + K)
    ph["platoon_split"] = ph["eb_ph"] - ph["eb_p"]
    ph["platoon_n"] = np.log1p(ph["n"])
    return ph[["pitcher_id", "batter_hand", "platoon_split", "platoon_n"]]


def apply_platoon_lookup(df, lookup):
    x = df.merge(lookup, on=["pitcher_id", "batter_hand"], how="left")
    x["platoon_split"] = x["platoon_split"].fillna(0.0)
    x["platoon_n"] = x["platoon_n"].fillna(0.0)
    return x


def matrix(df, cols, num_cols, med, cat_features):
    x = df[cols].copy()
    x[num_cols] = x[num_cols].fillna(med)
    for c in cat_features:
        x[c] = x[c].fillna("__NA__").astype(str)
    return x


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    t0 = time.time()
    raw = pd.read_csv(DATA_DIR / "train.csv")
    test_cols = pd.read_csv(DATA_DIR / "test.csv", nrows=0).columns.tolist()
    baseline_47 = [c for c in test_cols if c != ID]
    trend_cols = [f"trend_prev{k}" for k in PREV_PAIRS] + [f"trend_abs_prev{k}" for k in PREV_PAIRS]
    feature_cols = baseline_47 + trend_cols + ["platoon_split", "platoon_n",
                                               "count_state", "handedness_matchup"]
    num_cols = [c for c in feature_cols if c not in BASELINE_CATS]

    full = add_trend(raw)
    full = add_count_state(full)
    full = add_matchup(full)
    log(f"loaded train={full.shape}", t0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for val_season, monitor_season in TARGETS:
        log(f"\n=== val_season={val_season} (fit<{val_season}, 내부monitor={monitor_season}) ===", t0)

        fit_raw = full[full.season < val_season].copy()
        val_raw = full[full.season == val_season].copy()
        # league_avg: fit(=학습에 쓰는 시즌)만으로 계산 -- raw 전체(미래 포함) 금지
        league_avg = fit_raw[TARGET].mean()

        # 플래툰 lookup도 fit에서만 생성 (기존 v12와 동일 원칙)
        fit_feat = add_platoon_prior_cumulative(fit_raw, league_avg)
        lookup = build_platoon_lookup(fit_raw, league_avg)
        val_feat = apply_platoon_lookup(val_raw, lookup)

        med = fit_feat[num_cols].median(numeric_only=True)
        x_fit_full = matrix(fit_feat, feature_cols, num_cols, med, BASELINE_CATS)
        y_fit_full = fit_feat[TARGET]
        x_val = matrix(val_feat, feature_cols, num_cols, med, BASELINE_CATS)
        log(f"fit={len(fit_feat)} val={len(val_feat)}", t0)

        # ---- 내부 홀드아웃으로 best_iter 결정 (타깃시즌 라벨 미사용) ----
        internal_train_raw = fit_raw[fit_raw.season < monitor_season]
        internal_monitor_raw = fit_raw[fit_raw.season == monitor_season]
        if len(internal_train_raw) == 0 or len(internal_monitor_raw) == 0:
            log(f"내부 분할 불가(train={len(internal_train_raw)}, monitor={len(internal_monitor_raw)}) "
                f"-- MAX_ITER 고정으로 대체", t0)
            best_iters = [MAX_ITER] * len(SEEDS)
        else:
            it_fit = add_platoon_prior_cumulative(internal_train_raw, league_avg)
            it_lookup = build_platoon_lookup(internal_train_raw, league_avg)
            it_monitor = apply_platoon_lookup(internal_monitor_raw, it_lookup)
            x_it_fit = matrix(it_fit, feature_cols, num_cols, med, BASELINE_CATS)
            x_it_monitor = matrix(it_monitor, feature_cols, num_cols, med, BASELINE_CATS)
            best_iters = []
            for seed in SEEDS:
                params = dict(CB_PARAMS, iterations=MAX_ITER, random_seed=seed)
                m = CatBoostClassifier(**params)
                m.fit(x_it_fit, it_fit[TARGET], cat_features=BASELINE_CATS,
                      eval_set=(x_it_monitor, it_monitor[TARGET]), use_best_model=True)
                bi = m.get_best_iteration() or MAX_ITER
                best_iters.append(bi)
                log(f"  내부홀드아웃 seed={seed} best_iter={bi}", t0)
                del m

        # ---- 최종: fit 전체로 고정 iteration만큼 학습 (eval_set 없음, 타깃시즌 미사용) ----
        preds = []
        for seed, bi in zip(SEEDS, best_iters):
            params = dict(CB_PARAMS, iterations=bi, random_seed=seed)
            params.pop("od_type", None)
            params.pop("od_wait", None)  # 조기종료 자체를 안 씀 -- 고정 iteration
            m = CatBoostClassifier(**params)
            m.fit(x_fit_full, y_fit_full, cat_features=BASELINE_CATS)
            p = m.predict_proba(x_val)[:, 1]
            preds.append(p)
            log(f"  최종 seed={seed} iterations={bi} 학습완료", t0)
            del m

        ens = np.mean(np.vstack(preds), axis=0)
        out = pd.DataFrame({ID: val_raw[ID].to_numpy(), "pred": ens})
        out_path = OUT_DIR / f"hw_v12_{val_season}.csv"
        out.to_csv(out_path, index=False)
        log(f"저장: {out_path} ({len(out)}행, pred 평균={ens.mean():.4f})", t0)

    print(f"\n총 소요시간 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()

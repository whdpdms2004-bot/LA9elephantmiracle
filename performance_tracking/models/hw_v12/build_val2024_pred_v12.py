"""팀 결합용 공통 Val2024 예측 -- v12 기준으로 갱신
(규격: cowork/sj/claude/16_ENSEMBLE_HANDOFF.md).

★ 왜 갱신하나
기존 cowork/hw/val2024_pred.csv는 v9 시절 모델(Val2024 BSS 733.28)이다.
그 사이 hw 라인이 v10(실LB 892.12) -> v11(895.98) -> v12(912.13)로 올라갔는데
결합용 예측만 그대로였다. 그 결과 w*=M^-1 A로 팀 결합 가중치를 계산하면
hw 가중치가 -0.013으로 나와 사실상 버려진다 -- hw가 팀에서 가장 비상관인
조각(hw-sj 0.922, hw-yn 0.914 vs sj-yn 0.957)인데 낡은 예측 탓에 과소평가되는
상황이다.

규격 준수
- fit: season<2024만 (2024는 학습에 전혀 안 씀)
- val: season==2024 전체 253,507행, train.csv 원본 순서 그대로
- season/bucket offset 미적용 -- 2025 외삽용 보정이라 2024 자체 평가엔 안 맞음.
  raw 앙상블 확률을 낸다.

v12 구성 (train_best_model_v12.py PHASE1과 동일)
- feature 57개: anchor(baseline47) + trend6 + platoon2 + count_state
                + handedness_matchup
- cat_features 9개: top_bottom, game_type, base_state, pitcher_team_id,
  batter_team_id, count_state, pitcher_hand, batter_hand, handedness_matchup
- CatBoost 16-seed 배깅

출력: cowork/hw/val2024_pred.csv (row_id, control_success)

실행:
    py build_val2024_pred_v12.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # 저장소 루트/data
OUT_PATH = Path(__file__).resolve().parent / "val2024_pred.csv"

ID = "row_id"
TARGET = "control_success"
BASELINE_CATS = ["top_bottom", "game_type", "base_state",
                 "pitcher_team_id", "batter_team_id", "count_state",
                 "pitcher_hand", "batter_hand", "handedness_matchup"]
PREV_PAIRS = (1, 3, 5)
K_PLATOON = 300.0
SEEDS = list(range(2026, 2026 + 16))

CB_PARAMS = dict(
    loss_function="Logloss", eval_metric="BrierScore", depth=6,
    learning_rate=0.03, l2_leaf_reg=25, random_strength=0.6, border_count=128,
    thread_count=-1, grow_policy="Depthwise", boosting_type="Plain",
    bootstrap_type="Bernoulli", subsample=0.7, rsm=0.7,
    verbose=False, od_type="Iter", od_wait=100, allow_writing_files=False,
    iterations=1500,
)


def log(msg, t0):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def score(y, p):
    y = np.asarray(y)
    brier = float(np.mean((y - p) ** 2))
    r = y.mean()
    base = r * (1 - r)
    return brier, (max(0.0, 100000 * (1 - brier / base)) if base > 0 else 0.0)


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
    x["handedness_matchup"] = (x["pitcher_hand"].astype(str) + "_"
                               + x["batter_hand"].astype(str))
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
    feature_cols = (baseline_47 + trend_cols + ["platoon_split", "platoon_n"]
                    + ["count_state", "handedness_matchup"])
    num_cols = [c for c in feature_cols if c not in BASELINE_CATS]
    league_avg = raw[TARGET].mean()
    full = add_trend(raw)
    full = add_count_state(full)
    full = add_matchup(full)
    log(f"loaded train={full.shape}", t0)

    fit_raw = full[full.season < 2024].copy()
    val_raw = full[full.season == 2024].copy()  # 원본 순서 그대로 유지 (재정렬 없음)
    log(f"fit(2019-2023)={len(fit_raw)}, val(2024)={len(val_raw)} (스펙: 253,507)", t0)
    assert len(val_raw) == 253507, f"val 행 수가 스펙과 다름: {len(val_raw)}"

    fit = add_platoon_prior_cumulative(fit_raw, league_avg)
    lookup = build_platoon_lookup(fit_raw, league_avg)  # fit만으로 생성 (2024 미사용)
    val = apply_platoon_lookup(val_raw, lookup)

    med = fit[num_cols].median(numeric_only=True)
    x_fit = fit[feature_cols].copy()
    x_fit[num_cols] = x_fit[num_cols].fillna(med)
    for c in BASELINE_CATS:
        x_fit[c] = x_fit[c].fillna("__NA__").astype(str)
    x_val = val[feature_cols].copy()
    x_val[num_cols] = x_val[num_cols].fillna(med)
    for c in BASELINE_CATS:
        x_val[c] = x_val[c].fillna("__NA__").astype(str)
    y_fit, y_val = fit[TARGET], val[TARGET]

    preds = []
    for i, seed in enumerate(SEEDS, 1):
        params = dict(CB_PARAMS)
        params["random_seed"] = seed
        model = CatBoostClassifier(**params)
        model.fit(x_fit, y_fit, cat_features=BASELINE_CATS, eval_set=(x_val, y_val), use_best_model=True)
        p = model.predict_proba(x_val)[:, 1]
        preds.append(p)
        log(f"seed {i}/{len(SEEDS)}={seed} done, best_iter={model.get_best_iteration()}", t0)

    ens = np.mean(np.vstack(preds), axis=0)
    brier, bss = score(y_val, ens)
    log(f"16-seed 앙상블 BSS={bss:.2f} (참고: 기존 공유분 v9시절 733.28)", t0)

    out = pd.DataFrame({ID: val_raw[ID].to_numpy(), TARGET: ens})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    log(f"저장 완료: {OUT_PATH} ({len(out)}행)", t0)
    print(f"\n총 소요시간 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()

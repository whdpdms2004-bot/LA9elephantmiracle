"""좌우(hand) 인코딩 실험 -- v10의 +34.72를 만든 것과 같은 유형의 변경.

배경: v10에서 pitcher_team_id/batter_team_id를 숫자 -> CatBoost 네이티브
범주형으로 바꿔서 실LB +34.72를 얻었다. 확인해보니 pitcher_hand/batter_hand도
**int64 2값(1/2)인데 지금 숫자로 처리되고 있다** -- team_id가 v10 전에 있던
상황과 정확히 같다. 아직 한 번도 안 건드린 부분.

추가로: 내 platoon_split은 "투수별" 좌우 성향을 EB 수축으로 잡는 피처인데,
**단순 좌우 매치업(L/L, L/R, R/L, R/R) 자체를 범주형으로 준 적이 없다.**
yn님은 handedness_matchup으로 이걸 쓰고 있다.

이번 세션에서 확인된 패턴:
    통함  -> "있는 정보를 어떻게 인코딩하느냐" 변경
             (배깅 +50.76, 구간오프셋 +8.84, team_id 범주형 +34.72)
    실패  -> 피처 추가/제거/사후보정
             (sj preprocess_lab 원자 4개 전부 -, isotonic 전부 -,
              drift-drop 붕괴, 찬우 파생 10개 중 1개 빼고 전부 -)
이번 실험은 전자에 해당한다.

arm 4개, v11 베이스(anchor+trend+platoon+team_id범주형+count_state) 위에서
결정 fold(2024) 단일시드 스크리닝:
    baseline      : v11 그대로
    A_hand_cat    : pitcher_hand/batter_hand를 범주형으로 등록
    B_matchup     : handedness_matchup(4레벨 범주형) 추가
    C_both        : A + B

정직한(fit<2024, val==2024), train.csv만, 리더보드 미참조.

실행:
    py validate_hand_encoding.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # 저장소 루트/data
ID = "row_id"
TARGET = "control_success"
V11_CATS = ["top_bottom", "game_type", "base_state",
            "pitcher_team_id", "batter_team_id", "count_state"]
PREV_PAIRS = (1, 3, 5)
K_PLATOON = 300.0
VAL_SEASON = 2024

CB_PARAMS = dict(
    loss_function="Logloss", eval_metric="BrierScore", depth=6,
    learning_rate=0.03, l2_leaf_reg=25, random_strength=0.6, border_count=128,
    thread_count=-1, grow_policy="Depthwise", boosting_type="Plain",
    bootstrap_type="Bernoulli", subsample=0.7, rsm=0.7,
    verbose=False, od_type="Iter", od_wait=100, allow_writing_files=False,
    iterations=1500, random_seed=2026,
)


def log(msg, t0):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def score(y, p):
    y = np.asarray(y)
    brier = float(np.mean((y - p) ** 2))
    r = y.mean()
    base = r * (1 - r)
    return max(0.0, 100000 * (1 - brier / base)) if base > 0 else 0.0


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
    """좌우 매치업 4레벨. 원본 컬럼 재조합일 뿐 새 정보 없음."""
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


def matrix(df, cols, num_cols, med, cat_features):
    x = df[cols].copy()
    fill = [c for c in num_cols if c in x.columns]
    x[fill] = x[fill].fillna(med[fill])
    for c in cat_features:
        if c in x.columns:
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
    platoon_cols = ["platoon_split", "platoon_n"]
    v11_cols = baseline_47 + trend_cols + platoon_cols + ["count_state"]
    league_avg = raw[TARGET].mean()
    log(f"loaded train={raw.shape}, v11 피처={len(v11_cols)}", t0)

    full = add_trend(raw)
    full = add_count_state(full)
    full = add_matchup(full)

    fit_raw = full[full.season < VAL_SEASON].copy()
    val_raw = full[full.season == VAL_SEASON].copy()
    fit_p = add_platoon_prior_cumulative(fit_raw, league_avg)
    lookup_p = build_platoon_lookup(fit_raw, league_avg)
    val_p = apply_platoon_lookup(val_raw, lookup_p)

    # (추가 피처, 추가 범주형 등록)
    ARMS = {
        "baseline(v11)":  ([], []),
        "A_hand_cat":     ([], ["pitcher_hand", "batter_hand"]),
        "B_matchup":      (["handedness_matchup"], ["handedness_matchup"]),
        "C_both":         (["handedness_matchup"],
                           ["pitcher_hand", "batter_hand", "handedness_matchup"]),
    }

    results = {}
    for label, (extra_cols, extra_cats) in ARMS.items():
        cols = v11_cols + extra_cols
        cat_features = V11_CATS + extra_cats
        num_cols = [c for c in cols if c not in cat_features]
        med = fit_p[num_cols].median(numeric_only=True)
        x_fit = matrix(fit_p, cols, num_cols, med, cat_features)
        x_val = matrix(val_p, cols, num_cols, med, cat_features)

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(x_fit, fit_p[TARGET], cat_features=cat_features,
                  eval_set=(x_val, val_p[TARGET]), use_best_model=True)
        p = model.predict_proba(x_val)[:, 1]
        bss = score(val_p[TARGET], p)
        results[label] = bss
        log(f"[{label}] BSS={bss:8.2f} best_iter={model.get_best_iteration()} "
            f"(피처 {len(cols)}, 범주형 {len(cat_features)})", t0)

    print("\n" + "=" * 78)
    print(f"좌우(hand) 인코딩 결과 (val={VAL_SEASON} 결정 fold)")
    print("=" * 78)
    base = results["baseline(v11)"]
    for label, bss in results.items():
        d = bss - base
        flag = "  <- 개선" if label != "baseline(v11)" and d > 3 else ""
        print(f"  {label:18s} BSS={bss:8.2f}  Δ={d:+7.2f}{flag}")
    print("\n  (판정 기준: V61 원칙대로 내부 +3 미만은 제출 근거로 안 씀)")

    print(f"\n총 소요시간 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()

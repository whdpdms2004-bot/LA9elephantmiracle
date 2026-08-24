"""drift-drop 이식 -- 학습 시즌 간 분포이동이 큰 피처를 자동 제거.

sj님 신규 기법(cowork/sj, 커밋 d720fb3/1f20738). 학습 시즌 중 최근 두 해를
골라 수치 피처별로 분포이동을 재고, 큰 순으로 N개를 뺀다. **평가 데이터는
일절 보지 않는다** (학습 시즌 둘만 비교) -- 완전히 시간순/행독립 준수.

sj님 실측(reverse 타깃, 111+피처 중 65개 제거가 봉우리):
    제거 0개  f23 762.5  f24 993.9
    제거 65   f23 860.1  f24 1015.7   <- 지표=quantile 기준 최고
두 fold가 같이 오르고 시드 3개에서 안정(853.0/860.1/842.0).

지표는 분위수 기반이 평균차보다 나음 -- "평균은 그대로인데 분포 모양만
바뀌는 피처"를 평균차 지표는 놓치기 때문. 여기서도 quantile을 기본으로 쓰되
mean도 같이 재서 비교한다.

왜 내 라인에 중요한가: 이번 세션 최대 고질병이 "로컬 검증은 강한데 실LB 전이가
안 됨"(플래툰 로컬 +82 -> 실LB +0.7)이었는데, 그게 정확히 분포이동 문제다.
구조변경은 내 라인에서 항상 잘 전이됐다(배깅 +50.76, 구간오프셋 +8.84,
team_id 범주형 +34.72).

주의 -- sj님과 규모가 다르다:
    sj: 111+ 피처 중 65개(약 58%) 제거가 봉우리
    나: 56 피처(baseline47 + trend6 + platoon2 + count_state)
같은 비율이면 ~32개인데, 내 피처는 이미 선별된 것들이라 봉우리가 훨씬
앞쪽일 수 있다. N을 0~30까지 넓게 훑는다.

또 하나 -- 내 오프셋은 추론 시 test 원본 컬럼 asof_pitcher_n으로 구간을
고르는데, 이건 "모델 피처로 쓰는 것"과 별개다. 피처에서 빠져도 오프셋은
그대로 동작한다.

범주형(top_bottom, game_type, base_state, team_id 2개, count_state)은
드롭 후보에서 제외 -- sj님도 수치 피처만 대상으로 함.

v11 베이스(anchor+trend+platoon+team_id범주형+count_state) 위에서,
결정 fold(2024) 단일시드로 N을 훑는다. 승자는 별도로 2023까지 확인.

실행:
    py validate_drift_drop.py
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
BASELINE_CATS = ["top_bottom", "game_type", "base_state",
                 "pitcher_team_id", "batter_team_id", "count_state"]
PREV_PAIRS = (1, 3, 5)
K_PLATOON = 300.0
VAL_SEASON = 2024  # 결정 fold
DROP_SWEEP = [0, 5, 10, 15, 20, 25, 30]

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


def compute_drift(fit_df, num_cols, season_a, season_b, metric="quantile"):
    """학습 시즌 두 개(a, b)만 비교해서 피처별 분포이동 점수를 낸다.
    평가 데이터는 일절 참조하지 않음."""
    a = fit_df[fit_df.season == season_a]
    b = fit_df[fit_df.season == season_b]
    scores = {}
    for c in num_cols:
        va = a[c].to_numpy(dtype=float)
        vb = b[c].to_numpy(dtype=float)
        va = va[~np.isnan(va)]
        vb = vb[~np.isnan(vb)]
        if len(va) < 100 or len(vb) < 100:
            scores[c] = 0.0
            continue
        pooled_sd = np.std(np.concatenate([va, vb]))
        if pooled_sd <= 0 or not np.isfinite(pooled_sd):
            scores[c] = 0.0
            continue
        if metric == "mean":
            s = abs(va.mean() - vb.mean()) / pooled_sd
        elif metric == "quantile":
            qs = np.arange(0.1, 1.0, 0.1)  # 십분위
            qa = np.quantile(va, qs)
            qb = np.quantile(vb, qs)
            s = float(np.median(np.abs(qa - qb))) / pooled_sd
        else:
            raise ValueError(metric)
        scores[c] = float(s) if np.isfinite(s) else 0.0
    return pd.Series(scores).sort_values(ascending=False)


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
    all_feature_cols = baseline_47 + trend_cols + platoon_cols + ["count_state"]
    num_cols_all = [c for c in all_feature_cols if c not in BASELINE_CATS]
    league_avg = raw[TARGET].mean()
    log(f"loaded train={raw.shape}, 전체피처={len(all_feature_cols)} "
        f"(수치 {len(num_cols_all)}, 범주 {len(BASELINE_CATS)})", t0)

    full = add_trend(raw)
    full = add_count_state(full)

    fit_raw = full[full.season < VAL_SEASON].copy()
    val_raw = full[full.season == VAL_SEASON].copy()
    fit_p = add_platoon_prior_cumulative(fit_raw, league_avg)
    lookup_p = build_platoon_lookup(fit_raw, league_avg)
    val_p = apply_platoon_lookup(val_raw, lookup_p)

    # 드리프트: 학습 구간의 최근 두 시즌만 비교 (val_season은 절대 안 봄)
    fit_seasons = sorted(fit_p.season.unique())
    season_a, season_b = fit_seasons[-2], fit_seasons[-1]
    log(f"드리프트 비교 시즌: {season_a} vs {season_b} (val={VAL_SEASON}은 미참조)", t0)

    drift_q = compute_drift(fit_p, num_cols_all, season_a, season_b, metric="quantile")
    drift_m = compute_drift(fit_p, num_cols_all, season_a, season_b, metric="mean")

    print("\n" + "=" * 78)
    print("분포이동 상위 20개 (quantile 지표 기준)")
    print("=" * 78)
    print(f"  {'피처':42s} {'quantile':>10s} {'mean':>10s}")
    for c in drift_q.index[:20]:
        print(f"  {c:42s} {drift_q[c]:>10.4f} {drift_m.get(c, 0):>10.4f}")

    results = {}
    for n_drop in DROP_SWEEP:
        dropped = list(drift_q.index[:n_drop])
        cols = [c for c in all_feature_cols if c not in dropped]
        num_cols = [c for c in cols if c not in BASELINE_CATS]
        med = fit_p[num_cols].median(numeric_only=True)
        x_fit = matrix(fit_p, cols, num_cols, med, BASELINE_CATS)
        x_val = matrix(val_p, cols, num_cols, med, BASELINE_CATS)

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(x_fit, fit_p[TARGET], cat_features=BASELINE_CATS,
                  eval_set=(x_val, val_p[TARGET]), use_best_model=True)
        p = model.predict_proba(x_val)[:, 1]
        bss = score(val_p[TARGET], p)
        results[n_drop] = bss
        log(f"drop={n_drop:3d} (남은피처 {len(cols):2d}) BSS={bss:8.2f} "
            f"best_iter={model.get_best_iteration()}", t0)

    print("\n" + "=" * 78)
    print(f"drift-drop 스윕 결과 (val={VAL_SEASON} 결정 fold, quantile 지표)")
    print("=" * 78)
    base = results[0]
    best_n = max(results, key=results.get)
    for n_drop in DROP_SWEEP:
        d = results[n_drop] - base
        flag = "  <- 봉우리" if n_drop == best_n and n_drop != 0 else ""
        print(f"  drop={n_drop:3d}  BSS={results[n_drop]:8.2f}  Δ={d:+7.2f}{flag}")
    print(f"\n  최고: drop={best_n} (Δ{results[best_n]-base:+.2f})")
    if best_n > 0:
        print(f"  제거된 피처: {list(drift_q.index[:best_n])}")

    print(f"\n총 소요시간 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()

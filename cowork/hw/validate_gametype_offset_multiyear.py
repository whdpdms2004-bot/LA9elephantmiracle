"""새 실험(결과: 기각): hw x sj 구간별 결합 분석(hw_sj_segmented_blend_analysis.md)에서
나온 발견 -- game_type=F에서 모델 간 상관이 0.748로 뚝 떨어짐 -- 이 "결합 가치"뿐
아니라 hw 자신의 모델 캘리브레이션 편향에도 해당하는지, 그리고 연도 간 재현되는지
확인.

결론: 재현 안 됨. F-R 편향 gap의 부호가 연도마다 뒤집힘 (2022 음수, 2023 강한
양수, 2024 혼재). game_type을 오프셋의 새 축으로 쓰면 안 됨 -- 이미 보류시킨
오프셋 5구간/스무딩(투수표본 축, 2022 vs 2024 불일치)과 같은 실패 패턴.
2023 fold는 학습이 best_iter=3에서 멈춰서(조기종료) 그 해 자체가 이상치일
가능성도 있음 (isotonic 캘리브레이션 실험에서도 2023이 레짐변화 이상해로 확인됨).

참고: `submit_teamblend.zip`이 "팀범주화"(구간별 결합)를 쓰는데, 그 구간이
game_type과 비슷한 축이라면 같은 함정(연도 특정 노이즈에 과적합)에 빠질 수
있어서 팀에 공유용으로 이 결과를 올림.

leave-future-out(honest) 방식: fit은 val_season 이전 연도만, val은 해당 연도.
리더보드 미참조, train.csv만 사용 (RULES.md §2 준수).

실행 (저장소 루트에서, 수 분~수십 분 소요 -- fold별 CatBoost 1회 학습):
    py cowork/hw/validate_gametype_offset_multiyear.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
ID = "row_id"
TARGET = "control_success"
BASELINE_CATS = ["top_bottom", "game_type", "base_state"]
PREV_PAIRS = (1, 3, 5)
K_PLATOON = 300.0
VAL_SEASONS = [2022, 2023, 2024]
N_EDGES = [-1, 100, 2000, 1e9]
N_LABELS = ["<100", "100-2000", "2000+"]

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


def add_trend(df):
    x = df.copy()
    for k in PREV_PAIRS:
        recent = f"asof_pitcher_prev{k}_game_success_rate"
        x[f"trend_prev{k}"] = x[recent] - x["asof_pitcher_success_rate"]
        x[f"trend_abs_prev{k}"] = x[f"trend_prev{k}"].abs()
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
    platoon_cols = ["platoon_split", "platoon_n"]
    cols = baseline_47 + trend_cols + platoon_cols
    league_avg = raw[TARGET].mean()
    full = add_trend(raw)
    log(f"loaded train={full.shape}", t0)

    all_rows = []
    for val_season in VAL_SEASONS:
        fit_raw = full[full.season < val_season].copy()
        val_raw = full[full.season == val_season].copy()
        if len(fit_raw) == 0:
            log(f"val={val_season} 스킵 (fit 데이터 없음, 최초 연도)", t0)
            continue

        fit = add_platoon_prior_cumulative(fit_raw, league_avg)
        lookup = build_platoon_lookup(fit_raw, league_avg)
        val = apply_platoon_lookup(val_raw, lookup)

        num_cols = [c for c in cols if c not in BASELINE_CATS]
        med = fit[num_cols].median(numeric_only=True)
        x_fit = fit[cols].copy()
        x_fit[num_cols] = x_fit[num_cols].fillna(med)
        for c in BASELINE_CATS:
            x_fit[c] = x_fit[c].fillna("__NA__").astype(str)
        x_val = val[cols].copy()
        x_val[num_cols] = x_val[num_cols].fillna(med)
        for c in BASELINE_CATS:
            x_val[c] = x_val[c].fillna("__NA__").astype(str)

        model = CatBoostClassifier(**CB_PARAMS)
        model.fit(x_fit, fit[TARGET], cat_features=BASELINE_CATS,
                  eval_set=(x_val, val[TARGET]), use_best_model=True)
        val = val.copy()
        val["pred"] = model.predict_proba(x_val)[:, 1]
        log(f"val={val_season} 학습 완료 (best_iter={model.get_best_iteration()})", t0)

        val["n_bucket"] = pd.cut(val["asof_pitcher_n"].fillna(0), N_EDGES, labels=N_LABELS)
        for gt in ["R", "F"]:
            for nb in N_LABELS:
                sub = val[(val["game_type"] == gt) & (val["n_bucket"] == nb)]
                if len(sub) == 0:
                    continue
                pm, am = sub["pred"].mean(), sub[TARGET].mean()
                all_rows.append(dict(val_season=val_season, game_type=gt, n_bucket=nb,
                                      n=len(sub), pred_mean=pm, actual_mean=am,
                                      bias_pp=(pm - am) * 100))

    res = pd.DataFrame(all_rows)
    print("\n" + "=" * 78)
    print("game_type x 투수표본구간별 편향 -- 연도별 재현성 확인")
    print("=" * 78)
    for val_season in VAL_SEASONS:
        sub = res[res.val_season == val_season]
        if len(sub) == 0:
            continue
        print(f"\n[val={val_season}]")
        for nb in N_LABELS:
            r_row = sub[(sub.game_type == "R") & (sub.n_bucket == nb)]
            f_row = sub[(sub.game_type == "F") & (sub.n_bucket == nb)]
            if len(r_row) == 0 or len(f_row) == 0:
                continue
            r_bias = r_row.bias_pp.iloc[0]
            f_bias = f_row.bias_pp.iloc[0]
            gap = f_bias - r_bias
            print(f"  n={nb:10s}  R편향={r_bias:+7.3f}%p (n={int(r_row.n.iloc[0]):6d})  "
                  f"F편향={f_bias:+7.3f}%p (n={int(f_row.n.iloc[0]):6d})  F-R gap={gap:+7.3f}%p")

    print("\n" + "=" * 78)
    print("판정: 모든 연도 x 모든 n구간에서 F-R gap이 같은 부호(F가 더 과대예측)면")
    print("      -> game_type을 오프셋의 새 축으로 추가할 근거 있음 (재현성 확보)")
    print("      부호가 뒤집히는 연도/구간이 있으면 -> 2024 한정 노이즈, 채택 보류")
    print("=" * 78)
    print(f"\n총 소요시간 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()

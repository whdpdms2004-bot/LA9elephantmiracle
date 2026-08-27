# -*- coding: utf-8 -*-
"""[ye 재현] `cowork/ye/champion_structural_improvement.ipynb` 의 모델을 되살린다.

## 왜 이걸 하나

`ye` 는 팀에서 **완전히 독립된 파이프라인**이다 — platoon/hand 중심 CatBoost,
walk-forward 다시즌 폴드, 우리와 다른 피처 생성. 우리 세 모델(cw·sj·hw)은 전부
ρ 0.89~0.96 으로 묶여 있어서, ye 가 **팀 최대 미확인 자산**이다.

그런데 저장소에 산출물이 없다 — 노트북 출력은 지워졌고 모델 파일은 맥 로컬
경로(`/Users/joyeeun/...`)다. "팀원에게 요청할 것" 으로 미뤄뒀는데,
**노트북에 코드가 전부 있으므로 직접 재현하면 된다.**

## 원본과 같게 한 것

노트북 셀 3 의 피처 엔지니어링을 그대로 옮겼다.

    K_SMOOTH=20 로 n-쌍 rate 평활 · prev-game rate 결측 채움
    hand_prior (pitcher_id x batter_hand as-of 성공률, K_HAND_SMOOTH=20)
    trend/form_change 6+3열 · recent_volatility · score_situation · li_log
    pitch_mix_entropy · RISP/만루 상호작용 7열
    CatBoostClassifier(iterations=300, lr=0.05, depth=6, Logloss)
    folds = [(seasons[:i], seasons[i])]   <- walk-forward. 우리 폴드와 같다

## 원본과 다르게 한 것 — 더 엄격하게

원본은 `global_means` 를 **train 전체**(검증 시즌 포함)에서 뽑는다. 평활 상수라
누수는 미미하지만, 이 산출물의 용도가 **정직한 OOF** 이므로 폴드마다
학습 시즌에서만 뽑는다. 원본보다 엄격한 방향이다.

`hand_prior` 는 그룹 내 누적(cumcount/cumsum)이라 그 행 **이전** 만 본다 —
노트북 1.5/1.6 절이 CSV 순서가 그룹 내 시간순임을 검증해뒀다. 그대로 쓴다.

## 산출물

    performance_tracking/val/ye_hand_2024.csv
    performance_tracking/val/ye_hand_2022.csv

이게 있어야 §31 앙상블에 ye 를 멤버로 넣어 판정할 수 있다.

    python ye_repro.py --seeds 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
ROOT = FINAL.parents[2]
PT = ROOT / "performance_tracking"

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

K_SMOOTH = 20
K_HAND_SMOOTH = 20
N_PAIRED = {
    "asof_pitcher_n": ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                       "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
                       "asof_pitcher_strike_rate"],
    "asof_batter_n": ["asof_batter_success_rate", "asof_batter_middle_rate"],
    "asof_pitcher_pitchmix_n": ["asof_pitcher_fastball_rate",
                                "asof_pitcher_breaking_rate",
                                "asof_pitcher_offspeed_rate"],
}
NO_N = ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate"]
CAT_COLS = ["top_bottom", "game_type", "base_state", "score_situation"]
CAT_MODEL = ["top_bottom", "base_state", "score_situation"]
CAT_PARAMS = dict(iterations=300, learning_rate=0.05, depth=6)


def log(m):
    print(m, flush=True)


def bss(p, y):
    r = y.mean()
    return 100000.0 * (1.0 - ((p - y) ** 2).mean() / (r * (1.0 - r)))


def build(df, fit_mask):
    """노트북 셀3 의 피처 엔지니어링. `global_means` 만 fit_mask 에서 뽑는다."""
    d = df.copy()
    for n_col, rate_cols in N_PAIRED.items():
        n = d[n_col]
        for rc in rate_cols:
            gm = d.loc[fit_mask, rc].mean()
            d[rc + "_smoothed"] = (n * d[rc].fillna(0) + K_SMOOTH * gm) / (n + K_SMOOTH)
    for c in NO_N:
        d[c + "_filled"] = d[c].fillna(d.loc[fit_mask, c].mean())

    g = d.groupby(["pitcher_id", "batter_hand"])["control_success"]
    cnt = g.cumcount()
    ssum = g.cumsum() - d["control_success"]
    prior = (ssum / cnt).fillna(0)
    d["asof_pitcher_vs_hand_success_rate_smoothed"] = (
        cnt * prior + K_HAND_SMOOTH * d["asof_pitcher_success_rate_smoothed"]
    ) / (cnt + K_HAND_SMOOTH)

    for a, b in ((1, 5), (1, 3), (3, 5)):
        d["trend_success_%d_%d" % (a, b)] = (
            d["asof_pitcher_prev%d_game_success_rate_filled" % a]
            - d["asof_pitcher_prev%d_game_success_rate_filled" % b])
        d["trend_middle_%d_%d" % (a, b)] = (
            d["asof_pitcher_prev%d_game_middle_rate_filled" % a]
            - d["asof_pitcher_prev%d_game_middle_rate_filled" % b])
    for k in (1, 3, 5):
        d["form_change_%d" % k] = (d["asof_pitcher_prev%d_game_success_rate_filled" % k]
                                   - d["asof_pitcher_success_rate_smoothed"])
    d["recent_volatility"] = d[["asof_pitcher_prev%d_game_success_rate_filled" % k
                                for k in (1, 3, 5)]].std(axis=1)
    d["score_situation"] = np.select(
        [d["score_diff_pitcher_team"] == 0, d["score_diff_pitcher_team"].abs() <= 3,
         d["score_diff_pitcher_team"] > 3], ["동점", "접전", "대승"], default="대패")
    d["li_log"] = np.log1p(d["li"])
    d["pitcher_win_expectancy"] = np.where(d["top_bottom"] == "T",
                                           d["home_win_expectancy"],
                                           d["away_win_expectancy"])
    pr = d[["asof_pitcher_%s_rate_smoothed" % k
            for k in ("fastball", "breaking", "offspeed")]].to_numpy()
    prn = pr / pr.sum(axis=1, keepdims=True)
    d["pitch_mix_entropy"] = -(prn * np.log(prn + 1e-10)).sum(axis=1)

    risp = ((d["runner_on_2b"] == 1) | (d["runner_on_3b"] == 1)).astype(int)
    loaded = ((d["runner_on_1b"] == 1) & (d["runner_on_2b"] == 1)
              & (d["runner_on_3b"] == 1)).astype(int)
    d["risp_x_success"] = risp * d["asof_pitcher_success_rate_smoothed"]
    d["bases_loaded_x_li"] = loaded * d["li_log"]
    d["runner3b_x_scorediff"] = d["runner_on_3b"] * d["score_diff_pitcher_team"]
    d["bases_loaded_x_balls"] = loaded * d["balls_before"]
    d["risp_x_strikes"] = risp * d["strikes_before"]
    d["force_in_situation"] = ((loaded == 1) & (d["balls_before"] == 3)).astype(int)
    d["no_runners"] = (d["num_runners_on"] == 0).astype(int)

    raw = sum(N_PAIRED.values(), []) + NO_N
    excl = (["row_id", "control_success", "season", "game_dayofweek", "game_month",
             "game_type", "pitcher_team_id", "batter_team_id"] + raw
            + ["li", "home_win_expectancy", "away_win_expectancy"])
    feats = [c for c in d.columns if c not in excl and not c.startswith("game_type_")]
    # 원본은 `pd.get_dummies(columns=cat_cols)` 로 범주 원본열을 **제거**한 뒤
    # `feature_cols_cat` 에서 더미를 걸러내고 `cat_cols_model` 을 다시 붙인다.
    # 여기서는 get_dummies 를 쓰지 않으므로(CatBoost 가 범주를 직접 다룬다)
    # 원본열과 더미 접두 둘 다 뺀 뒤 붙여야 이름이 겹치지 않는다.
    feats = [c for c in feats
             if c not in CAT_COLS
             and not any(c.startswith(cc + "_") for cc in CAT_COLS)]
    feats += CAT_MODEL
    for c in CAT_MODEL:
        d[c] = d[c].astype(str)
    return d, feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    import catboost as cb

    log("=" * 78)
    log("[ye 재현] champion_structural_improvement.ipynb 의 M0_HAND_CHAMPION")
    log("=" * 78)
    t0 = time.time()
    df = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    df.columns = [c.strip("﻿") for c in df.columns]
    log("  train %s행 · %d열  (%.0f초)" % (f"{len(df):,}", df.shape[1], time.time() - t0))

    for fold in (2024, 2022):
        fit = df["season"] <= fold - 1
        va = df["season"] == fold
        t1 = time.time()
        d, feats = build(df, fit)
        log("\nfold%d  학습 %s행 · 검증 %s행 · 피처 %d개  (전처리 %.0f초)"
            % (fold, f"{int(fit.sum()):,}", f"{int(va.sum()):,}", len(feats),
               time.time() - t1))
        Xt, yt = d.loc[fit, feats], d.loc[fit, "control_success"]
        Xv, yv = d.loc[va, feats], d.loc[va, "control_success"].to_numpy(float)
        acc = np.zeros(int(va.sum()))
        for sd in range(a.seeds):
            ts = time.time()
            m = cb.CatBoostClassifier(**CAT_PARAMS, random_state=42 + sd,
                                      loss_function="Logloss", eval_metric="AUC",
                                      verbose=False, allow_writing_files=False)
            m.fit(Xt, yt, cat_features=CAT_MODEL)
            p = m.predict_proba(Xv)[:, 1]
            acc += p
            log("    seed%d  %.0f초  단독 %.1f" % (sd, time.time() - ts, bss(p, yv)))
            del m
        p = acc / a.seeds
        log("  %d시드 평균  BSS %.1f  (평균예측 %.4f · 실제 %.4f)"
            % (a.seeds, bss(p, yv), p.mean(), yv.mean()))
        out = PT / "val" / ("ye_hand_%d.csv" % fold)
        pd.DataFrame({"row_id": d.loc[va, "row_id"].to_numpy(), "pred": p}).to_csv(
            out, index=False)
        log("  → %s" % out.name)
        del d, Xt, Xv
    log("\n총 %.1f분" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

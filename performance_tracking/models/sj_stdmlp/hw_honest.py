# -*- coding: utf-8 -*-
"""[hw 재생산] `hw_v12` 의 val 예측을 **정직하게** 다시 만든다 + val2022 를 채운다.

## 왜

두 가지가 겹쳤다.

**1. 공유된 예측이 오염돼 있다.**
`performance_tracking/models/hw_v12/build_val2024_pred_v12.py` 가

    model.fit(x_fit, y_fit, ..., eval_set=(x_val, y_val), use_best_model=True)

로 **검증 폴드의 라벨을 조기 종료에 쓴다.** `cowork/sj/three_way/NEXT_PLAN.md` 가
정확히 이걸 경고했다 — "정직한이란 채점 fold 를 조기 종료(eval_set)에 쓰지
않았다는 뜻이다. 저장된 예측 중 301개가 그 문제를 갖고 있다."

그 파일을 나는 "정직한 OOF" 로 믿고 팀 결합 판정에 써왔다.

**2. val2022 가 없다.** 규칙 1 의 비하락 관문을 hw 에 적용할 수 없었다.
"팀원에게 요청할 것" 으로 미뤄뒀는데 **학습 코드가 저장소에 있다.**

## 원본과 같게 / 다르게

같게: 피처 57개(baseline47 + trend6 + platoon2 + count_state + handedness_matchup),
범주 9개, CatBoost 파라미터, platoon 룩업을 fit 에서만 만드는 것.

**다르게 — 두 곳:**

1. `eval_set` 을 **쓰지 않는다.** 고정 반복(`iterations`)으로 끝까지 학습한다.
   조기 종료가 없으면 `use_best_model` 도 의미가 없다.
2. `league_avg` 를 **fit 행에서만** 뽑는다 (원본은 train 전체 = 검증 시즌 포함).

둘 다 원본보다 엄격한 방향이다. 결과는 원본보다 낮게 나오는 것이 정상이고,
그게 **정직한 값**이다.

## 폴드

    fold 2024   fit season<2024   -> val season==2024
    fold 2022   fit season<2022   -> val season==2022

    python hw_honest.py --seeds 4 --iters 1200
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

ID = "row_id"
TARGET = "control_success"
CATS = ["top_bottom", "game_type", "base_state", "pitcher_team_id",
        "batter_team_id", "count_state", "pitcher_hand", "batter_hand",
        "handedness_matchup"]
PREV = (1, 3, 5)
K_PLATOON = 300.0
CB = dict(loss_function="Logloss", eval_metric="BrierScore", depth=6,
          learning_rate=0.03, l2_leaf_reg=25, random_strength=0.6,
          border_count=128, thread_count=-1, grow_policy="Depthwise",
          boosting_type="Plain", bootstrap_type="Bernoulli", subsample=0.7,
          rsm=0.7, verbose=False, allow_writing_files=False)


def log(m):
    print(m, flush=True)


def bss(y, p):
    r = y.mean()
    return 100000.0 * (1.0 - ((y - p) ** 2).mean() / (r * (1.0 - r)))


def prep(df):
    x = df.copy()
    for k in PREV:
        x["trend_prev%d" % k] = (x["asof_pitcher_prev%d_game_success_rate" % k]
                                 - x["asof_pitcher_success_rate"])
        x["trend_abs_prev%d" % k] = x["trend_prev%d" % k].abs()
    x["count_state"] = (x["balls_before"].astype(str) + "-"
                        + x["strikes_before"].astype(str))
    x["handedness_matchup"] = (x["pitcher_hand"].astype(str) + "_"
                               + x["batter_hand"].astype(str))
    return x


def platoon_cum(x, avg, K=K_PLATOON):
    d = x.copy()
    s_ph = d.groupby(["pitcher_id", "batter_hand"])[TARGET].cumsum() - d[TARGET]
    n_ph = d.groupby(["pitcher_id", "batter_hand"]).cumcount()
    s_p = d.groupby("pitcher_id")[TARGET].cumsum() - d[TARGET]
    n_p = d.groupby("pitcher_id").cumcount()
    d["platoon_split"] = ((s_ph + K * avg) / (n_ph + K)
                          - (s_p + K * avg) / (n_p + K))
    d["platoon_n"] = np.log1p(n_ph)
    return d


def platoon_lookup(src, avg, K=K_PLATOON):
    ph = src.groupby(["pitcher_id", "batter_hand"])[TARGET].agg(
        n="count", s="sum").reset_index()
    p = src.groupby("pitcher_id")[TARGET].agg(p_n="count", p_s="sum").reset_index()
    ph = ph.merge(p, on="pitcher_id", how="left")
    ph["platoon_split"] = ((ph["s"] + K * avg) / (ph["n"] + K)
                           - (ph["p_s"] + K * avg) / (ph["p_n"] + K))
    ph["platoon_n"] = np.log1p(ph["n"])
    return ph[["pitcher_id", "batter_hand", "platoon_split", "platoon_n"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=1200)
    a = ap.parse_args()
    from catboost import CatBoostClassifier

    t0 = time.time()
    raw = pd.read_csv(ROOT / "data" / "train.csv")
    test_cols = pd.read_csv(ROOT / "data" / "test.csv", nrows=0).columns.tolist()
    base47 = [c for c in test_cols if c != ID]
    feats = (base47 + ["trend_prev%d" % k for k in PREV]
             + ["trend_abs_prev%d" % k for k in PREV]
             + ["platoon_split", "platoon_n", "count_state", "handedness_matchup"])
    num = [c for c in feats if c not in CATS]
    full = prep(raw)
    log("=" * 78)
    log("[hw 재생산] eval_set 없이 · league_avg 를 fit 에서만 · %d시드 x %d반복"
        % (a.seeds, a.iters))
    log("=" * 78)
    log("  train %s행 · 피처 %d개 (범주 %d)" % (f"{len(full):,}", len(feats), len(CATS)))

    for fold in (2024, 2022):
        fr = full[full.season < fold].copy()
        vr = full[full.season == fold].copy()
        # ★ league_avg 를 fit 행에서만 (원본은 train 전체)
        avg = fr[TARGET].mean()
        fit = platoon_cum(fr, avg)
        val = vr.merge(platoon_lookup(fr, avg), on=["pitcher_id", "batter_hand"],
                       how="left")
        val["platoon_split"] = val["platoon_split"].fillna(0.0)
        val["platoon_n"] = val["platoon_n"].fillna(0.0)
        med = fit[num].median(numeric_only=True)
        xf, xv = fit[feats].copy(), val[feats].copy()
        xf[num] = xf[num].fillna(med)
        xv[num] = xv[num].fillna(med)
        for c in CATS:
            xf[c] = xf[c].fillna("__NA__").astype(str)
            xv[c] = xv[c].fillna("__NA__").astype(str)
        yf = fit[TARGET]
        yv = val[TARGET].to_numpy(float)
        log("\nfold%d  fit %s행 · val %s행 · league_avg %.6f"
            % (fold, f"{len(xf):,}", f"{len(xv):,}", avg))
        acc = np.zeros(len(xv))
        for i in range(a.seeds):
            ts = time.time()
            m = CatBoostClassifier(**CB, iterations=a.iters, random_seed=2026 + i)
            # ★ eval_set 을 쓰지 않는다 — 검증 라벨로 조기 종료하면 오염이다
            m.fit(xf, yf, cat_features=CATS)
            p = m.predict_proba(xv)[:, 1]
            acc += p
            log("    seed%d %.0f초  단독 %.1f" % (i, time.time() - ts, bss(yv, p)))
            del m
        p = acc / a.seeds
        log("  %d시드 평균 BSS %.1f  (평균예측 %.4f · 실제 %.4f)"
            % (a.seeds, bss(yv, p), p.mean(), yv.mean()))
        out = PT / "val" / ("hw_v12_honest_%d.csv" % fold)
        pd.DataFrame({"row_id": vr[ID].to_numpy(), "pred": p}).to_csv(out, index=False)
        log("  → %s" % out.name)
        if fold == 2024:
            old = pd.read_csv(PT / "val" / "hw_v12_2024.csv")
            om = old.set_index("row_id").loc[vr[ID].to_numpy()]["pred"].to_numpy()
            log("  ★공유분(eval_set 오염) %.1f  vs  정직 %.1f   차 %+.1f · ρ %.4f"
                % (bss(yv, om), bss(yv, p), bss(yv, p) - bss(yv, om),
                   np.corrcoef(om, p)[0, 1]))
    log("\n총 %.1f분" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
